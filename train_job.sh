#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/finetune_%j.out

source ~/.bashrc
conda activate ai-test-py312
cd ~/Model-RPG
python PEFT-Finetuning.py
