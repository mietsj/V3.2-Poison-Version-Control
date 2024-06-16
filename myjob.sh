#!/bin/bash
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=myjob-%j.out
#SBATCH --error=myjob-%j.err

python thesis-diffusion.py
