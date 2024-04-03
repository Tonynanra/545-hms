#!/bin/bash

#SBATCH --job-name=kaggle_preprocessing
#SBATCH --mail-user=ashtsang@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00-07:00:00
#SBATCH --account=eecs545w24_class
#SBATCH --partition=standard
#SBATCH --output=./out/%x-%j.out

eval "$(conda shell.bash hook)"
conda activate eecs545_proj

srun python kaggle_preprocessing.py
