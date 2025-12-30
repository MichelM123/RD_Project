# RD_Project
# Group 3 – EPIC-KITCHENS-100 Action Recognition (3D ResNet-18)

This repository contains the complete codebase used by **Group 3** for the EPIC-KITCHENS-100 action recognition project.  
The task is addressed using a **visual-only** approach based on a **3D ResNet-18** architecture pretrained on **Kinetics-400**, with separate prediction heads for **verbs (97 classes)** and **nouns (300 classes)**.

The repository also includes the full **Codabench validation and submission pipeline**

## 1. Project structure

The repository is organized as follows:

RD_Project/
├── data/
│   ├── annotations/
│   │   ├── EPIC_100_train.csv
│   │   └── EPIC_100_validation.csv
│   │
│   └── videos_640x360/
│       ├── P01/
│       │   ├── P01_01.MP4
│       │   └── ...
│       ├── P02/
│       │   └── ...
│       └── ...
│
├── notebooks/
│   └── (exploratory and analysis notebooks)
│
├── slurm/
│   ├── train.slurm          # SLURM script for training on VSC
│   └── val_ftta.slurm       # SLURM script for validation / inference with flip TTA
│
├── src/
│   ├── dataset.py                   # EPIC-KITCHENS dataset loading and preprocessing
│   ├── train.py                     # Training loop and optimization logic
│   ├── model_3d.py                  # 3D ResNet-18 model definition (dual-head)
│   ├── model.py                     # 2D ResNet-50 model (not used in this project's final submission)
│   ├── make_submission_res3d_fliptta.py  # Validation inference with flip TTA
│   ├── ensemble_checkpoints.py      # Logit-space ensembling of multiple checkpoints
│   ├── bias.py                      # Logit bias calibration (α = 0.3 on verb class 0)
│   ├── check_val.py                 # final submission file check
│   └── test_dataset.py              # Dataset checks and debugging
│
├── .gitignore
└── README.md

Code, data, and results are clearly separated to support reproducibility and ease of use.

## 2. Environment and dependencies

The code was tested with:
- PyTorch and torchvision (with video models enabled)
- CUDA-enabled GPU
- OpenCV (cv2)
- NumPy
- Pandas

Example installation:
  pip install torch torchvision numpy pandas opencv-python

On the Vlaams Supercomputer (VSC), the project is intended to be run using the cluster-provided CUDA-enabled PyTorch environment and SLURM job submission.

## 3. Dataset setup (EPIC-KITCHENS-100)

The EPIC-KITCHENS-100 dataset must be organized as follows:

<ROOT>/
  annotations/
    EPIC_100_train.csv
    EPIC_100_validation.csv
  videos_640x360/
    P01/
      P01_01.MP4
      ...
    P02/
      ...
    ...

Notes:
- Both .MP4 and .mp4 file extensions are supported.
- During training, annotation entries whose video files are missing are automatically filtered out.
- During validation inference, missing or unreadable videos are handled by generating black clips and logging coverage statistics.

## 4. Training (VSC / SLURM)

### 4.1 Launch training

Training is performed using train.py and submitted to the VSC using the provided SLURM script:

  sbatch train.slurm

Key training characteristics:
- Backbone: 3D ResNet-18 pretrained on Kinetics-400
- Input: single clip per action (16 frames, 160×160)
- Batch size: 8
- Loss: L = L_verb + 1.5 · L_noun
- Optimizer: AdamW with weight decay
- Learning-rate schedule: Cosine annealing
- End-to-end fine-tuning of backbone and heads

The best checkpoint is selected using a composite internal validation score (mean of verb top-1, noun top-1, and action top-1 accuracy).


### 4.2 Continuing training (resume)

Due to wall-time limits on the VSC, training is continued by resuming from saved checkpoints.

Workflow:
1. Identify the latest or best checkpoint in checkpoints/
2. Update train.slurm to resume from that checkpoint
3. Resubmit the job:
   sbatch train.slurm

This process is repeated until training is stopped.  

## 5. Validation and Codabench submission pipeline

Final evaluation is performed using the official EPIC-KITCHENS-100 Codabench server.  
The submission pipeline consists of the following steps.


### 5.1 Generate predictions with flip TTA

Script: make_submission_res3d_fliptta.py

This script:
- Uses EPIC_100_validation.csv
- Samples a single 16-frame clip per action
- Applies the same preprocessing as training
- Runs inference on:
  - the original clip
  - a horizontally flipped version of the clip
- Averages predictions in logit space

Typical execution (inside a SLURM job via val_ftta.slurm):
  sbatch val_ftta.slurm

Or directly:
  python make_submission_res3d_fliptta.py \
    --root <ROOT> \
    --checkpoint checkpoints/3d_best_epoch_v3_10.pt \
    --output preds/preds_epoch10_fliptta.pt \
    --batch_size 8 \
    --num_frames 16 \
    --spatial_size 160

This step is executed for three checkpoints corresponding to epochs 5, 8, and 10, producing:
  preds/preds_epoch5_fliptta.pt
  preds/preds_epoch8_fliptta.pt
  preds/preds_epoch10_fliptta.pt

### 5.2 Ensemble checkpoints

Script: ensemble_checkpoints.py

This script loads the three prediction files and performs equal-weight averaging of verb and noun logits:

  python ensemble_checkpoints.py

Output:
  submission_ens_5_8_10.pt

### 5.3 Apply logit bias calibration

Script: bias.py

This script applies a calibrated logit shift to verb class 0:
- Bias parameter: α = 0.3 (tuned empirically)
- Applied uniformly across all samples
- Performed directly in logit space

Run:
  python bias.py

Output:
  submission.pt

The same approach could be extended to other verb or noun classes, but only verb class 0 is biased in the final submission.

### 5.4 Codabench submission

The final file to upload to Codabench is:
  submission.pt

Codabench results are used as the authoritative measure of generalization performance, as internal validation metrics tend to overestimate accuracy due to the random split used during training.

## 6. Reproducibility notes

- All post-processing (flip TTA, ensembling, biasing) is performed in logit space
- Inference uses single-clip evaluation only (no multi-clip aggregation)
- No audio, optical flow, or object-centric features are used
- No class-balanced sampling, gradient clipping, or mixed-precision training is applied


## 7. End-to-end reproduction checklist

1. Place EPIC-KITCHENS-100 data under <ROOT>
2. Train the model:
   sbatch train.slurm
3. Resume training as needed until completion
4. Run flip-TTA inference for checkpoints at epochs 5, 8, and 10
5. Ensemble predictions:
   python ensemble_checkpoints.py
6. Apply bias calibration:
   python bias.py
7. Upload submission.pt to Codabench



