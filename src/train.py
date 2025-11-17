import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from dataset import EpicKitchensDataset
from model import Epic2DResNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train EPIC-KITCHENS 2D ResNet model")

    parser.add_argument(
        "--root",
        type=str,
        default="/data/leuven/380/vsc38046/RD_Project/epic-kitchens-data",
        help="Dataset root (contains annotations/, videos_640x360/).",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to EPIC_100_train.csv (if None, uses default under root/annotations/).",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_epic")

    # NEW: debug subset
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="If set, use only the first N samples from the dataset (for debugging).",
    )

    return parser.parse_args()


def evaluate(model, loader, device):
    model.eval()

    all_v_true, all_v_pred, all_v_prob = [], [], []
    all_n_true, all_n_pred, all_n_prob = [], [], []

    with torch.no_grad():
        for clips, verb, noun, meta in loader:
            clips = clips.to(device)      # [B, C, T, H, W]
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

    # ---- Dataset ----
    ds = EpicKitchensDataset(
        root=args.root,
        csv_path=args.csv_path,
        num_frames=args.num_frames,
        transform=None,  # you already return [0,1] floats
    )

    # Label counts from FULL train CSV (before subsampling!)
    df = ds.df  # already filtered to existing videos
    num_verbs = int(df["verb_class"].max()) + 1
    num_nouns = int(df["noun_class"].max()) + 1

    # Optional: limit dataset size for debugging
    if args.max_samples is not None and args.max_samples < len(ds):
        ds = Subset(ds, list(range(args.max_samples)))
        print(f"Using only first {args.max_samples} samples for training/validation.")

    print(f"Dataset segments: {len(ds)}")
    print(f"Num verbs: {num_verbs}, Num nouns: {num_nouns}")

    # Train/val split
    val_len = int(len(ds) * args.val_split)
    train_len = len(ds) - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])

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

    # ---- Model, loss, optimizer ----
    model = Epic2DResNet(num_verbs=num_verbs, num_nouns=num_nouns, pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (clips, verb, noun, meta) in enumerate(train_loader):
            clips = clips.to(device)      # [B, C, T, H, W]
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

        # ---- Evaluation ----
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

        # checkpointing
        val_f1_avg = 0.5 * (v["f1_macro"] + n["f1_macro"])
        if val_f1_avg > best_val_f1:
            best_val_f1 = val_f1_avg
            ckpt_path = os.path.join(args.save_dir, f"best_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                },
                ckpt_path,
            )
            print(f"Saved new best checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
