import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from dataset import EpicKitchensDataset
from model import Epic2DResNet
from model_3d import Epic3DResNet


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
        default="2d",
        help="Model type: '2d' (ResNet-50 TSN) or '3d' (r3d_18).",
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
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
        default=112,
        help="Resize frames to spatial_size x spatial_size "
             "(e.g., 112 for 3D ResNet, 224 for 2D TSN).",
    )

    return parser.parse_args()


def evaluate(model, loader, device):
    """Run evaluation on a loader and compute metrics."""
    model.eval()

    all_v_true, all_v_pred, all_v_prob = [], [], []
    all_n_true, all_n_pred, all_n_prob = [], [], []

    with torch.no_grad():
        for clips, verb, noun, meta in loader:
            clips = clips.to(device)
            verb = verb.to(device)
            noun = noun.to(device)

            verb_logits, noun_logits = model(clips)

            v_pred = verb_logits.argmax(dim=1)
            n_pred = noun_logits.argmax(dim=1)

            v_prob = torch.softmax(verb_logits, dim=1)
            n_prob = torch.softmax(noun_logits, dim=1)

            all_v_true.append(verb.cpu())
            all_v_pred.append(v_pred.cpu())
            all_v_prob.append(v_prob.cpu())

            all_n_true.append(noun.cpu())
            all_n_pred.append(n_pred.cpu())
            all_n_prob.append(n_prob.cpu())

    all_v_true = torch.cat(all_v_true).numpy()
    all_v_pred = torch.cat(all_v_pred).numpy()
    all_v_prob = torch.cat(all_v_prob).numpy()

    all_n_true = torch.cat(all_n_true).numpy()
    all_n_pred = torch.cat(all_n_pred).numpy()
    all_n_prob = torch.cat(all_n_prob).numpy()

    # Verb metrics
    v_acc = accuracy_score(all_v_true, all_v_pred)
    v_prec, v_rec, v_f1, _ = precision_recall_fscore_support(
        all_v_true, all_v_pred, average="macro", zero_division=0
    )
    try:
        v_auc = roc_auc_score(
            all_v_true, all_v_prob, multi_class="ovr", average="macro"
        )
    except ValueError:
        v_auc = None

    # Noun metrics
    n_acc = accuracy_score(all_n_true, all_n_pred)
    n_prec, n_rec, n_f1, _ = precision_recall_fscore_support(
        all_n_true, all_n_pred, average="macro", zero_division=0
    )
    try:
        n_auc = roc_auc_score(
            all_n_true, all_n_prob, multi_class="ovr", average="macro"
        )
    except ValueError:
        n_auc = None

    metrics = {
        "verb": {
            "accuracy": v_acc,
            "precision_macro": v_prec,
            "recall_macro": v_rec,
            "f1_macro": v_f1,
            "auc_macro_ovr": v_auc,
        },
        "noun": {
            "accuracy": n_acc,
            "precision_macro": n_prec,
            "recall_macro": n_rec,
            "f1_macro": n_f1,
            "auc_macro_ovr": n_auc,
        },
    }
    return metrics


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    #  Resolve train CSV path 
    if args.train_csv is None:
        args.train_csv = os.path.join(
            args.root, "annotations", "EPIC_100_train.csv"
        )

    print(f"Using TRAIN CSV (for full dataset & split): {args.train_csv}")

    #  Build full dataset from TRAIN CSV 
    full_ds = EpicKitchensDataset(
        root=args.root,
        csv_path=args.train_csv,
        num_frames=args.num_frames,
        transform=None,
        spatial_size=args.spatial_size,
    )

    # restrict to first N samples for debugging
    if args.max_samples is not None and args.max_samples < len(full_ds):
        from torch.utils.data import Subset

        full_ds = Subset(full_ds, list(range(args.max_samples)))
        print(f"Using only first {args.max_samples} samples from full dataset.")

    # Access underlying df for label counts (handles Subset or plain dataset)
    df_source = full_ds.dataset.df if hasattr(full_ds, "dataset") else full_ds.df
    num_verbs = int(df_source["verb_class"].max()) + 1
    num_nouns = int(df_source["noun_class"].max()) + 1

    #  Random train/val split 
    val_len = int(len(full_ds) * args.val_split)
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])

    print(f"Train segments: {train_len}")
    print(f"Val segments:   {val_len}")
    print(f"Num verbs: {num_verbs}, Num nouns: {num_nouns}")
    
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

    #  Model 
    if args.model_type == "2d":
        print("Using 2D ResNet-50 TSN model.")
        model = Epic2DResNet(num_verbs=num_verbs, num_nouns=num_nouns, pretrained=True)
    else:
        print("Using 3D ResNet-18 video model.")
        model = Epic3DResNet(num_verbs=num_verbs, num_nouns=num_nouns, pretrained=True)

    model.to(device)

    #  Loss & Optimizer 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_f1 = 0.0

    #  Training loop 
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (clips, verb, noun, meta) in enumerate(train_loader):
            clips = clips.to(device)
            verb = verb.to(device)
            noun = noun.to(device)

            optimizer.zero_grad()
            verb_logits, noun_logits = model(clips)

            loss_verb = criterion(verb_logits, verb)
            loss_noun = criterion(noun_logits, noun)
            loss = loss_verb + loss_noun

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * clips.size(0)

            if batch_idx % 50 == 0:
                print(
                    f"[Epoch {epoch}] Batch {batch_idx}/{len(train_loader)} "
                    f"- loss: {loss.item():.4f}",
                    flush=True,
                )

        train_loss = running_loss / train_len

        #  Evaluation on val split 
        metrics = evaluate(model, val_loader, device)
        v = metrics["verb"]
        n = metrics["noun"]

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Train loss: {train_loss:.4f}")
        print(
            "Verb  - acc: {acc:.3f}, prec: {prec:.3f}, rec: {rec:.3f}, "
            "f1: {f1:.3f}, auc: {auc}".format(
                acc=v["accuracy"],
                prec=v["precision_macro"],
                rec=v["recall_macro"],
                f1=v["f1_macro"],
                auc=v["auc_macro_ovr"],
            )
        )
        print(
            "Noun  - acc: {acc:.3f}, prec: {prec:.3f}, rec: {rec:.3f}, "
            "f1: {f1:.3f}, auc: {auc}".format(
                acc=n["accuracy"],
                prec=n["precision_macro"],
                rec=n["recall_macro"],
                f1=n["f1_macro"],
                auc=n["auc_macro_ovr"],
            )
        )

        #  Checkpointing 
        val_f1_avg = 0.5 * (v["f1_macro"] + n["f1_macro"])
        if val_f1_avg > best_val_f1:
            best_val_f1 = val_f1_avg
            ckpt_path = os.path.join(
                args.save_dir, f"{args.model_type}_best_epoch_{epoch}.pt"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"Saved new best checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
