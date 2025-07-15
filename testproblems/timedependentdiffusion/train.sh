#!/bin/bash



#SBATCH --partition=volta-longrun

#SBATCH --job-name=TRAIN

#SBATCH --mail-user=j.h.m.prins@tue.nl

#SBATCH --output=log_train.txt    

#SBATCH --time=23:59:00

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=1

#SBATCH --gres=gpu:1

#SBATCH --mem=64GB



python train.py



exit 0