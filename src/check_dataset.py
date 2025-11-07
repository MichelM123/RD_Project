import os
from pathlib import Path
import pandas as pd
import cv2

#  FIXED PATHS 
DATASET_ROOT = Path("/data/leuven/380/vsc38046/RD_Project/epic-kitchens-data")
ANNOTATIONS_DIR = DATASET_ROOT / "annotations"
VIDEOS_DIR = DATASET_ROOT / "videos_640x360"
TRAIN_CSV = ANNOTATIONS_DIR / "EPIC_100_train.csv"

def main():
    print("=== EPIC-KITCHENS sanity check ===")
    print(f"DATASET_ROOT: {DATASET_ROOT}")

    #  Check basic structure 
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Train CSV not found at: {TRAIN_CSV}")
    if not VIDEOS_DIR.exists():
        raise FileNotFoundError(f"Video directory not found at: {VIDEOS_DIR}")

    #  Load annotations 
    df = pd.read_csv(TRAIN_CSV)
    print(f"\nLoaded {len(df):,} rows from training CSV")
    print("Columns:", list(df.columns))

    #  Scan available video files 
    available = {}
    participants = sorted([p for p in os.listdir(VIDEOS_DIR) if (VIDEOS_DIR / p).is_dir()])
    print(f"\nFound {len(participants)} participant dirs locally:")
    print("   ", participants[:10], "..." if len(participants) > 10 else "")

    for participant in participants:
        part_dir = VIDEOS_DIR / participant
        for fname in os.listdir(part_dir):
            if fname.endswith(".MP4"):
                video_id = fname[:-4]
                available[(participant, video_id)] = part_dir / fname

    print(f"Discovered {len(available):,} .MP4 files actually present on disk")

    if len(available) == 0:
        raise RuntimeError("No .MP4 files found. Make sure the iget download finished correctly.")

    #  Filter annotations to only existing videos 
    df_local = df[df.apply(lambda r: (r["participant_id"], r["video_id"]) in available, axis=1)]
    print(f"Rows that match available videos: {len(df_local):,}")

    if len(df_local) == 0:
        raise RuntimeError("No matching videos found. Check dataset integrity or file structure.")

    #  Pick one random valid clip 
    row = df_local.sample(1).iloc[0]
    clip_path = available[(row["participant_id"], row["video_id"])]

    print("\nChosen valid segment:")
    print(f"  narration_id:   {row['narration_id']}")
    print(f"  participant_id: {row['participant_id']}")
    print(f"  video_id:       {row['video_id']}")
    print(f"  verb:           {row['verb']} (class {row['verb_class']})")
    print(f"  noun:           {row['noun']} (class {row['noun_class']})")
    print(f"  start_frame:    {row['start_frame']}")
    print(f"  stop_frame:     {row['stop_frame']}")
    print(f"\nClip path: {clip_path}")

    #  decoding one frame 
    cap = cv2.VideoCapture(str(clip_path))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print("\n Could NOT read a frame — video might be incomplete or corrupted.")
    else:
        print("\n Successfully read a frame!")
        print("Frame shape (H, W, C):", frame.shape)
        snapshot = DATASET_ROOT / "sample_frame.jpg"
        cv2.imwrite(str(snapshot), frame)
        print("Saved preview frame to:", snapshot)

    print("\n=== Dataset sanity check complete ===")

if __name__ == "__main__":
    main()
