#!/bin/bash
#SBATCH --job-name=interpolate_small_big_5
#SBATCH --mem=32G
#SBATCH --gres=gpu:A6000:4
#SBATCH --output=.slurm_logs/interpolate_small_big_5.out
#SBATCH --time=01-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vmasti@andrew.cmu.edu



python src/interpolation/interpolation_studies/interpolate_small_big.py \
    --small_pretrained_model_name "EleutherAI/pythia-70m" \
    --large_pretrained_model_name "EleutherAI/pythia-410m" \
    --tasks "lambada_openai" "paloma_dolma-v1_5" \
    --task_columns "perplexity,none" "word_perplexity,none" \
    --num_interpolations 5 \
    --output_json "eval/interpolation_small_big.json" \
    --output_plot "plots/interpolation_small_big" \
    --skip_eval