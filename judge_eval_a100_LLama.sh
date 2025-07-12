#!/bin/bash
#SBATCH --job-name=judge_eval_70b
#SBATCH --output=logs/judge_eval_70b_%j.out
#SBATCH --error=logs/judge_eval_70b_%j.err
#SBATCH --partition=compute
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# === Run evaluation script using large model ===
python eval.py \
  --judge meta-llama/Llama-3.3-70B-Instruct \
  --input model-responses.csv

