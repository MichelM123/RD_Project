import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from model_r2plus1d import EpicR2Plus1DResNet


# ------------------------ Validation Dataset ------------------------ #

class EpicValDataset(Dataset):
    """
    Minimal EPIC-KITCHENS-100 validation dataset for Codabench submission.

    - Uses EPIC_100_validation.csv (no verb/noun labels).
    - For each row:
        * Locates video in videos_640x360/<participant_id>/<video_id>.MP4 (or .mp4).
        * Samples num_frames uniformly between start_frame and stop_frame.
        * Reads frames sequentially, resizes to spatial_size x spatial_size.
        * Returns normalized clip [C, T, H, W] and meta with narration_id, real_fraction.
    - If video file is missing/unreadable, returns black clip and real_fraction=0.0.
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

        print(f"[EpicValDataset] Total segments in CSV: {len(self.df)}")
        missing = sum(1 for _, vp in self.samples if vp is None)
        if missing > 0:
            print(f"[EpicValDataset] WARNING: {missing} segments have no video file; using black clips.")

    def __len__(self):
        return len(self.samples)

    def _get_clip_indices(self, start_frame: int, stop_frame: int) -> np.ndarray:
        length = stop_frame - start_frame + 1
        if length <= 0:
            return np.full(self.num_frames, start_frame, dtype=int)
        rel = np.linspace(0, length - 1, num=self.num_frames)
        rel = np.round(rel).astype(int)
        rel = np.clip(rel, 0, length - 1)
        return start_frame + rel

    def _load_clip_from_video(self, video_path: Path, frame_indices: np.ndarray):
        """
        Sequential frame reading with cache.
        Returns:
            clip: [T, H, W, C] uint8
            valid_mask: [T] bool (True where frame is real video)
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

        sorted_idx = np.sort(frame_indices)
        order = np.argsort(frame_indices)

        first_idx = int(sorted_idx[0])
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_idx)
        current_frame_idx = first_idx

        success, frame = cap.read()
        if not success or frame is None:
            cap.release()
            return clip, valid_mask

        frame_resized = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
        frame_cache = {current_frame_idx: frame_resized}

        target_max = int(sorted_idx[-1])
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
            else:
                # remains black
                pass

        return clip, valid_mask

    def _normalize_clip(self, clip: np.ndarray) -> torch.Tensor:
        """
        clip: [T, H, W, C] BGR uint8 -> [C, T, H, W] normalized
        """
        clip = clip[:, :, :, ::-1]  # BGR -> RGB
        clip = clip.astype("float32") / 255.0
        clip = np.transpose(clip, (3, 0, 1, 2))  # C, T, H, W

        mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)[:, None, None, None]
        std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)[:, None, None, None]
        clip = (clip - mean) / std

        return torch.from_numpy(clip)

    def __getitem__(self, idx: int):
        row_idx, video_path = self.samples[idx]
        row = self.df.iloc[row_idx]

        start_frame = int(row["start_frame"])
        stop_frame = int(row["stop_frame"])
        frame_indices = self._get_clip_indices(start_frame, stop_frame)

        clip_bgr, valid_mask = self._load_clip_from_video(video_path, frame_indices)
        clip = self._normalize_clip(clip_bgr)
        real_fraction = float(valid_mask.mean())

        verb_class = -1
        noun_class = -1

        meta = {
            "narration_id": row["narration_id"],
            "participant_id": row["participant_id"],
            "video_id": row["video_id"],
            "real_fraction": real_fraction,
        }

        return clip, torch.tensor(verb_class), torch.tensor(noun_class), meta


# ------------------------ Submission Script ------------------------ #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Codabench submission.pt for EPIC-KITCHENS validation set (R(2+1)D-18)."
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
        help="Path to EPIC_100_validation.csv (if None: <root>/annotations/EPIC_100_validation.csv).",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained R(2+1)D-18 checkpoint.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="submission.pt",
        help="Output filename.",
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
        default=0,
        help="Number of dataloader workers.",
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
        help="Spatial size (resize to spatial_size x spatial_size).",
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=5,
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

    # Dataset & DataLoader
    val_ds = EpicValDataset(
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

    # Model
    num_verbs = 97
    num_nouns = 300

    model = EpicR2Plus1DResNet(num_verbs=num_verbs, num_nouns=num_nouns)
    model.to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    print("Loaded model state_dict from checkpoint.")
    model.eval()

    # Inference + stats
    all_preds = []

    total_segments = len(val_ds)
    count_pure_black = 0
    count_all_real = 0
    count_mixed = 0
    sum_real_fraction = 0.0

    with torch.no_grad():
        for batch_idx, (clips, _, _, meta) in enumerate(val_loader):
            clips = clips.to(device, non_blocking=True)

            verb_logits, noun_logits = model(clips)

            batch_narration_ids = meta["narration_id"]
            batch_real_fraction = meta["real_fraction"]

            for i in range(clips.size(0)):
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
                    "verb_output": verb_logits[i].detach().cpu(),
                    "noun_output": noun_logits[i].detach().cpu(),
                }
                all_preds.append(pred)

            if (batch_idx + 1) % args.log_interval == 0:
                print(f"Processed {(batch_idx + 1) * clips.size(0)} segments...")

    # Coverage stats
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
    v = sample["verb_output"]
    n = sample["noun_output"]
    assert isinstance(sample["narration_id"], str)
    assert isinstance(v, torch.Tensor) and v.ndim == 1 and v.shape[0] == num_verbs
    assert isinstance(n, torch.Tensor) and n.ndim == 1 and n.shape[0] == num_nouns
    print("Sanity checks passed (shapes and counts).")

    torch.save(all_preds, args.output)
    print(f"Successfully saved predictions to {args.output}")


if __name__ == "__main__":
    main()
