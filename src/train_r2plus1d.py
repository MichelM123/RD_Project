import os
import argparse
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import EpicKitchensDataset, build_transform
from model_r2plus1d import EpicR2Plus1DResNet


# ------------------------ Utilities ------------------------ #

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    """
    Computes the top-k accuracies for the specified values of k.
    output: [B, C]
    target: [B]
    Returns: list of accuracies (floats in [0,1])
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # [B, maxk]
    pred = pred.t()  # [maxk, B]
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # [maxk, B]

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k / float(batch_size)).item())
    return res


# ------------------------ Training & Evaluation ------------------------ #

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion_verb: nn.Module,
    criterion_noun: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    noun_loss_weight: float = 1.5,
    epoch: int = 1,
    log_interval: int = 50,  
) -> float:
    """
    Train for one epoch.

    log_interval: how many batches between progress prints.
    """
    model.train()
    running_loss = 0.0
    num_samples = 0

    num_batches = len(train_loader)
    print(f"[Epoch {epoch}] Number of training batches: {num_batches}", flush=True)

    for batch_idx, (clips, verbs, nouns, _) in enumerate(train_loader):
        clips = clips.to(device, non_blocking=True)
        verbs = verbs.to(device, non_blocking=True)
        nouns = nouns.to(device, non_blocking=True)

        optimizer.zero_grad()

        verb_logits, noun_logits = model(clips)
        loss_verb = criterion_verb(verb_logits, verbs)
        loss_noun = criterion_noun(noun_logits, nouns)
        loss = loss_verb + noun_loss_weight * loss_noun

        loss.backward()
        optimizer.step()

        batch_size = clips.size(0)
        running_loss += loss.item() * batch_size
        num_samples += batch_size

        # ---- progress print ----
        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
            print(
                f"Epoch {epoch} "
                f"[{batch_idx + 1}/{num_batches}] "
                f"Loss: {loss.item():.4f}",
                flush=True, 
            )

    return running_loss / max(num_samples, 1)




@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
) -> Dict[str, Any]:
    model.eval()

    total = 0
    correct_verb1 = 0
    correct_verb5 = 0
    correct_noun1 = 0
    correct_noun5 = 0
    correct_action1 = 0

    for clips, verbs, nouns, _ in val_loader:
        clips = clips.to(device, non_blocking=True)
        verbs = verbs.to(device, non_blocking=True)
        nouns = nouns.to(device, non_blocking=True)

        verb_logits, noun_logits = model(clips)

        # Verb metrics
        v1, v5 = accuracy(verb_logits, verbs, topk=(1, 5))
        # Noun metrics
        n1, n5 = accuracy(noun_logits, nouns, topk=(1, 5))

        # Action top-1: both verb and noun correct in top-1
        _, v_pred = verb_logits.max(dim=1)
        _, n_pred = noun_logits.max(dim=1)
        action_correct = (v_pred == verbs) & (n_pred == nouns)

        batch_size = clips.size(0)
        total += batch_size

        correct_verb1 += v1 * batch_size
        correct_verb5 += v5 * batch_size
        correct_noun1 += n1 * batch_size
        correct_noun5 += n5 * batch_size
        correct_action1 += action_correct.float().sum().item()

    verb_top1 = correct_verb1 / max(total, 1)
    verb_top5 = correct_verb5 / max(total, 1)
    noun_top1 = correct_noun1 / max(total, 1)
    noun_top5 = correct_noun5 / max(total, 1)
    action_top1 = correct_action1 / max(total, 1)

    return {
        "verb_top1": verb_top1,
        "verb_top5": verb_top5,
        "noun_top1": noun_top1,
        "noun_top5": noun_top5,
        "action_top1": action_top1,
    }


# ------------------------ Main Training Script ------------------------ #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EPIC-KITCHENS model with R(2+1)D-18 backbone."
    )

    parser.add_argument(
        "--root",
        type=str,
        default="/data/leuven/380/vsc38046/RD_Project/epic-kitchens-data",
        help="Dataset root (contains annotations/ and videos_640x360/).",
    )

    parser.add_argument(
        "--train_csv",
        type=str,
        default=None,
        help="Path to EPIC_100_train.csv (if None: <root>/annotations/EPIC_100_train.csv).",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate.",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay (AdamW).",
    )

    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing for CE loss.",
    )

    parser.add_argument(
        "--noun_loss_weight",
        type=float,
        default=1.5,
        help="Weight for noun loss in total loss.",
    )

    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of data used for validation (random split).",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints_epic_r2p1d",
        help="Directory to save checkpoints.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from.",
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames per clip.",
    )

    parser.add_argument(
        "--spatial_size",
        type=int,
        default=160,
        help="Resize frames to spatial_size x spatial_size.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def save_checkpoint(state: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None, device="cuda"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    start_epoch = ckpt.get("epoch", 0) + 1

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    print(f"Loaded checkpoint '{path}' (epoch {ckpt.get('epoch', 'unknown')})")
    return start_epoch, ckpt.get("metrics", {})


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.train_csv is None:
        args.train_csv = os.path.join(args.root, "annotations", "EPIC_100_train.csv")

    # ---------------- Dataset & DataLoaders ---------------- #
    train_transform = build_transform(is_train=True, normalize=True)
    val_transform = build_transform(is_train=False, normalize=True)

    base_ds = EpicKitchensDataset(
        root=args.root,
        csv_path=args.train_csv,
        num_frames=args.num_frames,
        transform=None,  # we'll wrap with TransformedSubset below
        spatial_size=args.spatial_size,
        has_labels=True,
    )

    num_samples = len(base_ds)
    val_len = int(num_samples * args.val_split)
    train_len = num_samples - val_len

    indices = torch.randperm(num_samples).tolist()
    train_indices = indices[:train_len]
    val_indices = indices[train_len:]

    class TransformedSubset(Subset):
        def __init__(self, dataset, indices, transform):
            super().__init__(dataset, indices)
            self.transform = transform
            self.dataset = dataset

        def __getitem__(self, idx):
            clip, verb, noun, meta = super().__getitem__(idx)
            if self.transform is not None:
                clip = self.transform(clip)
            return clip, verb, noun, meta

    train_ds = TransformedSubset(base_ds, train_indices, train_transform)
    val_ds = TransformedSubset(base_ds, val_indices, val_transform)

    print(f"[Train dataset] {len(train_ds)} samples")
    print(f"[Val dataset]   {len(val_ds)} samples")

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

    # ---------------- Model, Loss, Optimizer, Scheduler ---------------- #
    num_verbs = 97
    num_nouns = 300

    model = EpicR2Plus1DResNet(num_verbs=num_verbs, num_nouns=num_nouns)
    model.to(device)

    criterion_verb = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion_noun = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    start_epoch = 1
    best_score = -1.0
    best_metrics = {}

    if args.resume is not None:
        start_epoch, best_metrics = load_checkpoint(
            args.resume, model, optimizer, scheduler, device=device
        )
        if "val_score" in best_metrics:
            best_score = best_metrics["val_score"]

    # ---------------- Training Loop ---------------- #
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion_verb,
            criterion_noun,
            optimizer,
            device,
            noun_loss_weight=args.noun_loss_weight,
            epoch=epoch,
            log_interval=50,   
        )


        metrics = evaluate(model, val_loader, device)
        v1, v5 = metrics["verb_top1"], metrics["verb_top5"]
        n1, n5 = metrics["noun_top1"], metrics["noun_top5"]
        a1 = metrics["action_top1"]

        val_score = (v1 + n1 + a1) / 3.0

        print(
            f"Train loss: {train_loss:.4f}  "
            f"Verb - top1: {v1:.3f}, top5: {v5:.3f}  "
            f"Noun - top1: {n1:.3f}, top5: {n5:.3f}  "
            f"Action - top1: {a1:.3f}"
        )

        # Step scheduler AFTER validation (per epoch)
        scheduler.step()

        # Save best checkpoint
        if val_score > best_score:
            best_score = val_score
            metrics["val_score"] = val_score
            ckpt_path = os.path.join(
                args.checkpoint_dir, f"r2p1d18_best_epoch_{epoch}.pt"
            )
            save_checkpoint(
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
            best_metrics = metrics
            print(f"New best model saved to {ckpt_path} (val_score={val_score:.4f})")

    print("\nTraining finished.")
    print("Best metrics:", best_metrics)


if __name__ == "__main__":
    main()
