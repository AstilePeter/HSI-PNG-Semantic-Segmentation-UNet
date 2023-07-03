#!/bin/bash
#
#SBATCH --job-name=seg_train  # Job name
#SBATCH --gres=gpu:a100:1       # Ask for 1 Nvida k40 GPU. Available options are: "k20", "k40"

python  train.py
