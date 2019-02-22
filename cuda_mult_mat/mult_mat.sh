#!/bin/sh
#BATCH --job-name="estimate_pi"
#SBATCH --output="estimate_pi.%j.out"
#SBATCH --partition=gpu-shared
#SBATCH --gres=gpu:k80:1
#SBATCH --ntasks-per-node=6
#SBATCH --export=ALL
#SBATCH -t 01:30:00


/usr/local/cuda/bin/nvprof ./mult_mat
