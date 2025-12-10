import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from model_3d import Epic3DResNet


# ------------------------ Single-clip validation dataset ------------------------ #

class EpicValSingleClipDataset(Dataset):
    """
    EPIC-KITCHENS-100 validation dataset for single-clip inference
    (one 16-frame clip per segment, similar to training).
    """

    def __init__(
        self,
        root: str,
        csv_path: str,
        num_frames: int = 16,
        spatial_size: int = 160,
    ):
        self.root = Path(root)
        self.csv_path = Path(csv_path)
        self.video_root = self.root / "videos_640x360"

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        if not self.video_root.exists():
            raise FileNotFoundError(f"Video root not found: {self.video_root}")

        self.df = pd.read_csv(self.csv_path)
        self.num_frames = num_frames
        self.spatial_size = spatial_size

        self.samples = []
        for idx, row in self.df.iterrows():
            part = row["participant_id"]
            vid = row["video_id"]

            cand1 = self.video_root / part / f"{vid}.MP4"
            cand2 = self.video_root / part / f"{vid}.mp4"

            if cand1.exists():
                video_path = cand1
            elif cand2.exists():
                video_path = cand2
            else:
                video_path = None

            self.samples.append((idx, video_path))

        print(f"[EpicValSingleClipDataset] Total segments in CSV: {len(self.df)}")
        missing = sum(1 for _, vp in self.samples if vp is None)
        if missing > 0:
            print(f"[EpicValSingleClipDataset] WARNING: {missing} segments have no video; using black clips.")

    def __len__(self):
        return len(self.samples)

    def _get_clip_indices_uniform(self, start_frame: int, stop_frame: int) -> np.ndarray:
        """
        Uniformly sample num_frames indices between [start_frame, stop_frame].
        This should match the training sampling strategy.
        """
        length = stop_frame - start_frame + 1
        if length <= 0:
            return np.full(self.num_frames, start_frame, dtype=int)

        rel_indices = np.linspace(0, length - 1, num=self.num_frames)
        rel_indices = np.round(rel_indices).astype(int)
        rel_indices = np.clip(rel_indices, 0, length - 1)
        frame_indices = start_frame + rel_indices
        return frame_indices

    def _load_clip_from_video(self, video_path: Path, frame_indices: np.ndarray):
        """
        Sequential frame reading with caching.
        Returns clip [T, H, W, C] uint8 and valid_mask [T] bool.
        """
        num_frames = len(frame_indices)
        H = W = self.spatial_size

        clip = np.zeros((num_frames, H, W, 3), dtype=np.uint8)
        valid_mask = np.zeros(num_frames, dtype=bool)

        if video_path is None:
            return clip, valid_mask

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            return clip, valid_mask

        sorted_indices = np.sort(frame_indices)
        first_idx = int(sorted_indices[0])
        target_max = int(sorted_indices[-1])

        cap.set(cv2.CAP_PROP_POS_FRAMES, first_idx)
        current_frame_idx = first_idx

        success, frame = cap.read()
        if not success or frame is None:
            cap.release()
            return clip, valid_mask

        frame_resized = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
        frame_cache = {current_frame_idx: frame_resized}

        while current_frame_idx < target_max:
            success, frame = cap.read()
            if not success or frame is None:
                break
            current_frame_idx += 1
            frame_resized = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
            frame_cache[current_frame_idx] = frame_resized

        cap.release()

        last_valid = None
        for pos, fidx in enumerate(frame_indices):
            fidx = int(fidx)
            if fidx in frame_cache:
                clip[pos] = frame_cache[fidx]
                last_valid = frame_cache[fidx]
                valid_mask[pos] = True
            elif last_valid is not None:
                clip[pos] = last_valid
                valid_mask[pos] = True

        return clip, valid_mask

    def _normalize_clip(self, clip: np.ndarray) -> torch.Tensor:
        """
        clip: [T, H, W, C] uint8 BGR.
        Returns [C, T, H, W] float32 normalized with Kinetics stats.
        """
        clip = clip[:, :, :, ::-1]  # BGR -> RGB
        clip = clip.astype("float32") / 255.0
        clip = np.transpose(clip, (3, 0, 1, 2))  # C,T,H,W

        mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)[:, None, None, None]
        std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)[:, None, None, None]
        clip = (clip - mean) / std

        return torch.from_numpy(clip)

    def __getitem__(self, idx: int):
        row_idx, video_path = self.samples[idx]
        row = self.df.iloc[row_idx]

        start_frame = int(row["start_frame"])
        stop_frame = int(row["stop_frame"])

        frame_indices = self._get_clip_indices_uniform(start_frame, stop_frame)
        clip_bgr, valid_mask = self._load_clip_from_video(video_path, frame_indices)
        clip = self._normalize_clip(clip_bgr)  # [C,T,H,W]

        real_fraction = float(valid_mask.mean())

        meta = {
            "narration_id": row["narration_id"],
            "participant_id": row["participant_id"],
            "video_id": row["video_id"],
            "real_fraction": real_fraction,
        }

        # dummy labels
        return clip, torch.tensor(-1), torch.tensor(-1), meta


# ------------------------ Submission script ------------------------ #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Codabench submission.pt for EPIC-KITCHENS validation set "
                    "(single-clip + horizontal flip TTA, Res3D)."
    )

    parser.add_argument(
        "--root",
        type=str,
        default="/data/leuven/380/vsc38046/RD_Project/epic-kitchens-data",
        help="Dataset root (contains annotations/ and videos_640x360/).",
    )

    parser.add_argument(
        "--val_csv",
        type=str,
        default=None,
        help="Path to EPIC_100_validation.csv (default: <root>/annotations/EPIC_100_validation.csv).",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the best model checkpoint (e.g., 3d_best_epoch_v3_10.pt).",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="submission_fliptta.pt",
        help="Filename for saved predictions.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )

    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of frames per clip (must match training).",
    )

    parser.add_argument(
        "--spatial_size",
        type=int,
        default=160,
        help="Resize frames to spatial_size x spatial_size.",
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=40,
        help="How many batches between progress prints.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.val_csv is None:
        args.val_csv = os.path.join(
            args.root, "annotations", "EPIC_100_validation.csv"
        )

    print(f"Using VALIDATION CSV: {args.val_csv}")
    print(f"Using checkpoint: {args.checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---------------- Dataset & loader ---------------- #
    val_ds = EpicValSingleClipDataset(
        root=args.root,
        csv_path=args.val_csv,
        num_frames=args.num_frames,
        spatial_size=args.spatial_size,
    )
    print(f"[Val dataset] Using {len(val_ds)} segments.")

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---------------- Model & checkpoint ---------------- #
    num_verbs = 97
    num_nouns = 300

    model = Epic3DResNet(num_verbs=num_verbs, num_nouns=num_nouns)
    model.to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    print("Loaded model state_dict from checkpoint.")
    model.eval()

    # ---------------- Inference with flip TTA ---------------- #
    all_preds = []

    total_segments = len(val_ds)
    sum_real_fraction = 0.0
    count_pure_black = 0
    count_all_real = 0
    count_mixed = 0

    with torch.no_grad():
        for batch_idx, (clip, _, _, meta) in enumerate(val_loader):
            # clip: [B, C, T, H, W]
            clip = clip.to(device, non_blocking=True)

            # original
            verb_logits_orig, noun_logits_orig = model(clip)

            # horizontally flipped: flip along width (last spatial dim)
            clip_flipped = torch.flip(clip, dims=[-1])
            verb_logits_flip, noun_logits_flip = model(clip_flipped)

            # average logits
            verb_logits = 0.5 * (verb_logits_orig + verb_logits_flip)
            noun_logits = 0.5 * (noun_logits_orig + noun_logits_flip)

            B = clip.size(0)
            batch_narration_ids = meta["narration_id"]
            batch_real_fraction = meta["real_fraction"]

            for i in range(B):
                rf = float(batch_real_fraction[i])
                sum_real_fraction += rf
                if rf == 0.0:
                    count_pure_black += 1
                elif rf == 1.0:
                    count_all_real += 1
                else:
                    count_mixed += 1

                pred = {
                    "narration_id": batch_narration_ids[i],
                    "verb_output": verb_logits[i].cpu(),
                    "noun_output": noun_logits[i].cpu(),
                }
                all_preds.append(pred)

            if (batch_idx + 1) % args.log_interval == 0:
                print(f"Processed {(batch_idx + 1) * B} segments...")

    avg_real_fraction = sum_real_fraction / float(total_segments)
    print("----- Video coverage stats -----")
    print(f"Total segments:           {total_segments}")
    print(f"Pure black segments:      {count_pure_black}")
    print(f"All-real segments:        {count_all_real}")
    print(f"Mixed (partially real):   {count_mixed}")
    print(f"Average real frame frac:  {avg_real_fraction:.4f}")

    print(f"Total predictions collected: {len(all_preds)}")
    assert len(all_preds) == len(val_ds)

    sample = all_preds[0]
    assert isinstance(sample["narration_id"], str)
    assert isinstance(sample["verb_output"], torch.Tensor) and sample["verb_output"].shape[0] == num_verbs
    assert isinstance(sample["noun_output"], torch.Tensor) and sample["noun_output"].shape[0] == num_nouns

    torch.save(all_preds, args.output)
    print(f"Successfully saved predictions to {args.output}")


if __name__ == "__main__":
    main()
