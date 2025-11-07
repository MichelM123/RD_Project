import os
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EpicKitchensDataset(Dataset):
    """
    EPIC-KITCHENS-100 dataset (downscaled 640x360 videos).

    Returns for each item:
        clip: FloatTensor [C, T, H, W] in [0, 1]
        verb_class: LongTensor scalar
        noun_class: LongTensor scalar
        meta: dict with narration_id, participant_id, video_id
    """

    def __init__(
        self,
        root: str = "/data/leuven/380/vsc38046/RD_Project/epic-kitchens-data",
        csv_path: Optional[str] = None,
        num_frames: int = 16,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            root: dataset root directory (where annotations/ and videos_640x360/ live)
            csv_path: path to annotation CSV (defaults to EPIC_100_train.csv)
            num_frames: number of frames to sample per clip
            transform: optional transform applied to the clip tensor [C, T, H, W]
        """
        self.root = Path(root)
        self.video_root = self.root / "videos_640x360"

        if csv_path is None:
            csv_path = self.root / "annotations" / "EPIC_100_train.csv"
        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        if not self.video_root.exists():
            raise FileNotFoundError(f"Video root not found: {self.video_root}")

        self.df = pd.read_csv(self.csv_path)
        self.num_frames = num_frames
        self.transform = transform

        # Filter to rows for which the video file actually exists (handles missing clips)
        keep_indices: List[int] = []
        for idx, row in self.df.iterrows():
            vpath = self._video_path(row)
            if vpath.exists():
                keep_indices.append(idx)

        if len(keep_indices) == 0:
            raise RuntimeError(
                "No annotation rows point to existing video files. "
                "Check that videos_640x360 contains .MP4 files."
            )

        self.df = self.df.loc[keep_indices].reset_index(drop=True)
        print(f"[EpicKitchensDataset] Using {len(self.df)} segments with existing videos.")

    def __len__(self) -> int:
        return len(self.df)

    def _video_path(self, row: pd.Series) -> Path:
        """
        Build the full video path. Handles .mp4 /.MP4 extensions.
        """
        part = row["participant_id"]
        vid = row["video_id"]

        # First try .MP4 (as seen on your system), then .mp4 for safety.
        cand1 = self.video_root / part / f"{vid}.MP4"
        cand2 = self.video_root / part / f"{vid}.mp4"

        if cand1.exists():
            return cand1
        if cand2.exists():
            return cand2

        # Fallback, even if it likely doesn't exist:
        return cand1

    def _sample_frame_indices(self, start_frame: int, stop_frame: int) -> np.ndarray:
        """
        Sample self.num_frames indices between [start_frame, stop_frame] inclusive.
        If the segment is shorter than num_frames, frames will be duplicated.
        """
        if stop_frame < start_frame:
            stop_frame = start_frame

        length = stop_frame - start_frame + 1
        if length <= 0:
            length = 1

        indices = np.linspace(0, length - 1, num=self.num_frames)
        indices = np.round(indices).astype(int)
        indices = np.clip(indices, 0, length - 1)
        return start_frame + indices

    def _load_clip(self, video_path: Path, frame_indices: np.ndarray) -> np.ndarray:
        """
        Load frames given absolute frame indices from a video.
        Returns numpy array [T, H, W, C] in uint8 (BGR).
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frames: List[np.ndarray] = []

        for fidx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
            ok, frame = cap.read()
            if not ok or frame is None:
                # If read fails, reuse last frame or use black frame
                if len(frames) > 0:
                    frame = frames[-1]
                else:
                    frame = np.zeros((360, 640, 3), dtype=np.uint8)
            frames.append(frame)

        cap.release()

        clip = np.stack(frames, axis=0)  # [T, H, W, C]
        return clip

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        video_path = self._video_path(row)
        start_frame = int(row["start_frame"])
        stop_frame = int(row["stop_frame"])

        frame_indices = self._sample_frame_indices(start_frame, stop_frame)
        clip_np = self._load_clip(video_path, frame_indices)  # [T, H, W, C], BGR uint8

        # BGR -> RGB
        clip_np = clip_np[:, :, :, ::-1]

        # To float32 [0, 1]
        clip_np = clip_np.astype("float32") / 255.0

        # [T, H, W, C] -> [C, T, H, W]
        clip_np = np.transpose(clip_np, (3, 0, 1, 2))

        clip = torch.from_numpy(clip_np)  # FloatTensor [C, T, H, W]

        if self.transform is not None:
            clip = self.transform(clip)

        verb_class = int(row["verb_class"])
        noun_class = int(row["noun_class"])

        meta: Dict[str, Any] = {
            "narration_id": row["narration_id"],
            "participant_id": row["participant_id"],
            "video_id": row["video_id"],
        }

        return clip, torch.tensor(verb_class), torch.tensor(noun_class), meta
