#!/bin/bash
#SBATCH --job-name=grow_small_large_random
#SBATCH --mem=32G
#SBATCH --gres=gpu:2080Ti:4
#SBATCH --output=.slurm_logs/grow_small_large_random.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu

nvidia-smi

# accelerate launch --num_processes=1

python src/interpolation/grow_small_large_diff.py