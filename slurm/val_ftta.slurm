#!/bin/bash -l
#SBATCH --account=lp_edu_rdlab
#SBATCH --clusters=wice
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=960:00
#SBATCH --job-name=project
#SBATCH --output=slurm-3d-submission-v2-%j.out


module load cluster/wice/interactive

conda activate rd_lab

cd /data/leuven/380/vsc38046/RD_Project/src

python -u make_submission_res3d_fliptta.py \
  --root /data/leuven/380/vsc38046/RD_Project/epic-kitchens-data \
  --checkpoint ./checkpoints_epic/3d_best_epoch_v3_10.pt \
  --output preds_epoch10_fliptta.pt \
  --batch_size 8 \
  --num_workers 4























