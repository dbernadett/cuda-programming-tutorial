#!/bin/sh
#BATCH --job-name="mult_mat"
#SBATCH --output="mult_mat.%j.out"
#SBATCH --partition=gpu-shared
#SBATCH --gres=gpu:k80:1
#SBATCH --ntasks-per-node=6
#SBATCH --export=ALL
#SBATCH -t 01:30:00


/usr/local/cuda/bin/nvprof ./mult_mat
