from torch.utils.data import DataLoader
from dataset import EpicKitchensDataset

def main():
    ds = EpicKitchensDataset(
        root="/data/leuven/380/vsc38046/RD_Project/epic-kitchens-data",
        num_frames=8,
    )

    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    for clips, verb, noun, meta in loader:
        print("Batch clips shape:", clips.shape)   # [B, C, T, H, W]
        print("Verb labels:", verb)
        print("Noun labels:", noun)

        # meta is a dict of lists: {'narration_id': [...], 'participant_id': [...], ...}
        print("Meta keys:", meta.keys())

        # Print meta info for the first sample in the batch:
        first_meta = {k: meta[k][0] for k in meta}
        print("Meta for first sample:", first_meta)

        break

if __name__ == "__main__":
    main()
