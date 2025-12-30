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
#SBATCH --output=slurm-3d-v3-%j.out


module load cluster/wice/interactive

conda activate rd_lab

cd /data/leuven/380/vsc38046/RD_Project/src

python train.py \
  --model_type 3d \
  --num_frames 16 \
  --spatial_size 160 \
  --epochs 25 \
  --batch_size 8 \



















