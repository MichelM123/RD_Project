import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import EpicKitchensDataset, build_transform
from model import Epic2DResNet
from model_3d import Epic3DResNet


# -------------------------------------------------------------------------
# Evaluation: Codabench-style accuracies only
# -------------------------------------------------------------------------


def evaluate(model, loader, device):
    """
    Evaluate model on given loader. Computes:

    - verb_top1, verb_top5
    - noun_top1, noun_top5
    - action_top1 (verb AND noun both correct)
    """
    model.eval()

    v_correct_top1 = 0
    v_correct_top5 = 0
    n_correct_top1 = 0
    n_correct_top5 = 0
    a_correct_top1 = 0
    total = 0

    with torch.no_grad():
        for clips, verb, noun, meta in loader:
            clips = clips.to(device)
            verb = verb.to(device)
            noun = noun.to(device)

            verb_logits, noun_logits = model(clips)
            batch_size = clips.size(0)
            total += batch_size

            # Verb top-1 / top-5
            v_top1 = verb_logits.argmax(dim=1)  # [B]
            v_top5 = verb_logits.topk(5, dim=1).indices  # [B, 5]

            v_correct_top1 += (v_top1 == verb).sum().item()
            v_correct_top5 += (v_top5 == verb.unsqueeze(1)).any(dim=1).sum().item()

            # Noun top-1 / top-5
            n_top1 = noun_logits.argmax(dim=1)
            n_top5 = noun_logits.topk(5, dim=1).indices

            n_correct_top1 += (n_top1 == noun).sum().item()
            n_correct_top5 += (n_top5 == noun.unsqueeze(1)).any(dim=1).sum().item()

            # Action top-1: both verb and noun must be correct in top-1
            a_correct_top1 += ((v_top1 == verb) & (n_top1 == noun)).sum().item()

    metrics = {
        "verb_top1": v_correct_top1 / total,
        "verb_top5": v_correct_top5 / total,
        "noun_top1": n_correct_top1 / total,
        "noun_top5": n_correct_top5 / total,
        "action_top1": a_correct_top1 / total,
    }
    return metrics


# -------------------------------------------------------------------------
# Small helper to wrap base dataset with transforms per split
# -------------------------------------------------------------------------


class TransformedSubset(torch.utils.data.Dataset):
    """
    Wrap a base dataset + a list of indices + a transform on the clip.
    base_ds: EpicKitchensDataset (with transform=None)
    indices: list of ints
    transform: callable applied to clip [C, T, H, W]
    """

    def __init__(self, base_ds, indices, transform):
        self.base_ds = base_ds
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        base_idx = self.indices[i]
        clip, verb, noun, meta = self.base_ds[base_idx]

        if self.transform is not None:
            clip = self.transform(clip)

        return clip, verb, noun, meta


# -------------------------------------------------------------------------
# Checkpoint loading (resume functionality)
# -------------------------------------------------------------------------


def load_checkpoint(ckpt_path, model, optimizer=None, scheduler=None, device="cuda"):
    """
    Load checkpoint and restore model, optimizer, (optionally) scheduler.
    Returns start_epoch = last_epoch + 1.
    """
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Restore model weights
    model.load_state_dict(ckpt["model_state_dict"])

    start_epoch = ckpt.get("epoch", 0) + 1
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'N/A')} -> resuming from epoch {start_epoch}")

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print("Restored optimizer state.")

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            print("Restored scheduler state.")
        except Exception as e:
            print(f"Could not restore scheduler state (will re-init schedule): {e}")

    return start_epoch


# -------------------------------------------------------------------------
# Argument parsing
# -------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EPIC-KITCHENS model (2D TSN or 3D ResNet-18)"
    )

    parser.add_argument(
        "--root",
        type=str,
        default="/data/leuven/380/vsc38046/RD_Project/epic-kitchens-data",
        help="Dataset root (contains annotations/, videos_640x360/).",
    )

    # Single CSV (train) we'll split it into train/val
    parser.add_argument(
        "--train_csv",
        type=str,
        default=None,
        help="Path to EPIC_100_train.csv "
             "(if None, uses root/annotations/EPIC_100_train.csv).",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        choices=["2d", "3d"],
        default="3d",
        help="Model type: '2d' (ResNet-50 TSN) or '3d' (r3d_18).",
    )

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of data (from train CSV) used for validation.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints_epic",
        help="Directory to store checkpoints.",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="If set, use only the first N samples (after filtering) "
             "from the full dataset (for debugging).",
    )

    parser.add_argument(
        "--spatial_size",
        type=int,
        default=160,
        help="Resize frames to spatial_size x spatial_size "
             "(e.g., 112 for 3D ResNet, 160/224 for higher-res nouns).",
    )

    parser.add_argument(
        "--noun_loss_weight",
        type=float,
        default=1.5,
        help="Relative weight for noun loss vs verb loss (alpha).",
    )

    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing for CrossEntropyLoss (0 disables).",
    )

    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a checkpoint (.pt) to resume training from.",
    )

    return parser.parse_args()


# -------------------------------------------------------------------------
# Main training
# -------------------------------------------------------------------------


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Resolve train CSV path
    if args.train_csv is None:
        args.train_csv = os.path.join(
            args.root, "annotations", "EPIC_100_train.csv"
        )

    print(f"Using TRAIN CSV (for full dataset & split): {args.train_csv}")

    # Build base dataset (no transform here; we add per-split transforms later)
    base_ds = EpicKitchensDataset(
        root=args.root,
        csv_path=args.train_csv,
        num_frames=args.num_frames,
        transform=None,
        spatial_size=args.spatial_size,
    )

    # Restrict to first N samples for debugging
    if args.max_samples is not None and args.max_samples < len(base_ds):
        from torch.utils.data import Subset

        base_ds = Subset(base_ds, list(range(args.max_samples)))
        print(f"Using only first {args.max_samples} samples from full dataset.")

        # When using Subset, the underlying dataset is base_ds.dataset
        df_source = base_ds.dataset.df
    else:
        df_source = base_ds.df

    # ------------------------------------------------------------------
    # OFFICIAL EPIC-KITCHENS-100 CLASS COUNTS (task definition)
    # ------------------------------------------------------------------
    # These are fixed by the benchmark, not by what happens to appear
    # in this particular train split.
    num_verbs = 97
    num_nouns = 300

    # For transparency: check what actually appears in *this* dataset
    max_verb_id = int(df_source["verb_class"].max())
    max_noun_id = int(df_source["noun_class"].max())
    print(f"Max verb id present in this dataset: {max_verb_id}")
    print(f"Max noun id present in this dataset: {max_noun_id}")

    if max_verb_id != num_verbs - 1:
        print(
            f"WARNING: Expected highest verb id {num_verbs - 1}, "
            f"but found {max_verb_id} in this train split."
        )

    if max_noun_id != num_nouns - 1:
        print(
            f"WARNING: Expected highest noun id {num_nouns - 1}, "
            f"but found {max_noun_id} in this train split. "
            "Some noun classes are unseen in training."
        )

    print(f"Num verbs (fixed): {num_verbs}, Num nouns (fixed): {num_nouns}")

    # Random train/val split (on indices)
    num_samples = len(base_ds)
    val_len = int(num_samples * args.val_split)
    train_len = num_samples - val_len

    indices = torch.randperm(num_samples).tolist()
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]

    print(f"Train segments: {train_len}")
    print(f"Val segments:   {val_len}")
    print(f"Num verbs: {num_verbs}, Num nouns: {num_nouns}")

    # Build transforms
    train_transform = build_transform(is_train=True, normalize=True)
    val_transform = build_transform(is_train=False, normalize=True)

    # Wrap base dataset with transforms per split
    train_ds = TransformedSubset(base_ds, train_indices, train_transform)
    val_ds = TransformedSubset(base_ds, val_indices, val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model
    if args.model_type == "2d":
        print("Using 2D ResNet-50 TSN model.")
        model = Epic2DResNet(num_verbs=num_verbs, num_nouns=num_nouns, pretrained=True)
    else:
        print("Using 3D ResNet-18 video model.")
        model = Epic3DResNet(
            num_verbs=num_verbs,
            num_nouns=num_nouns,
            pretrained=True,
            dropout=0.5,
        )

    model.to(device)

    # Loss & Optimizer
    if args.label_smoothing > 0.0:
        criterion_verb = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        criterion_noun = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        criterion_verb = nn.CrossEntropyLoss()
        criterion_noun = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Cosine LR scheduler across epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_val_score = 0.0

    # Resume logic
    start_epoch = 1
    if args.resume_from is not None:
        start_epoch = load_checkpoint(
            args.resume_from,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (clips, verb, noun, meta) in enumerate(train_loader):
            clips = clips.to(device)
            verb = verb.to(device)
            noun = noun.to(device)

            optimizer.zero_grad()
            verb_logits, noun_logits = model(clips)

            loss_verb = criterion_verb(verb_logits, verb)
            loss_noun = criterion_noun(noun_logits, noun)
            loss = loss_verb + args.noun_loss_weight * loss_noun

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * clips.size(0)

            if batch_idx % 50 == 0:
                print(
                    f"[Epoch {epoch}] Batch {batch_idx}/{len(train_loader)} "
                    f"- loss: {loss.item():.4f}",
                    flush=True,
                )

        train_loss = running_loss / len(train_ds)

        # Evaluation on val split
        metrics = evaluate(model, val_loader, device)
        v1, v5 = metrics["verb_top1"], metrics["verb_top5"]
        n1, n5 = metrics["noun_top1"], metrics["noun_top5"]
        a1 = metrics["action_top1"]

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Train loss: {train_loss:.4f}")
        print(
            "Verb   - top1: {v1:.3f}, top5: {v5:.3f}".format(
                v1=v1, v5=v5
            )
        )
        print(
            "Noun   - top1: {n1:.3f}, top5: {n5:.3f}".format(
                n1=n1, n5=n5
            )
        )
        print("Action - top1: {a1:.3f}".format(a1=a1))

        # Simple composite score for model selection
        val_score = (v1 + n1 + a1) / 3.0

        if val_score > best_val_score:
            best_val_score = val_score
            ckpt_path = os.path.join(
                args.save_dir, f"{args.model_type}_best_epoch_v3_{epoch}.pt"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "metrics": metrics,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"Saved new best checkpoint to {ckpt_path}")

        # Step LR scheduler
        scheduler.step()


if __name__ == "__main__":
    main()
