#!/bin/bash
#SBATCH --job-name=grow_gft_lpt_diff
#SBATCH --mem=32G
#SBATCH --gres=gpu:2080Ti:4
#SBATCH --output=.slurm_logs/grow_gft_lpt_diff.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu


python src/interpolation/grow_gft_lpt_diff.py