#!/bin/sh
#BATCH --job-name="add_vec"
#SBATCH --output="add_vec.%j.out"
#SBATCH --partition=gpu-shared
#SBATCH --gres=gpu:k80:1
#SBATCH --ntasks-per-node=6
#SBATCH --export=ALL
#SBATCH -t 01:30:00


/usr/local/cuda/bin/nvprof ./add_vec
