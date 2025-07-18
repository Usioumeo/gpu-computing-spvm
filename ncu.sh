#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=test
#SBATCH --output=test-%j.out
#SBATCH --error=test-%j.err
module load CUDA/12.5

sudo $(which ncu) --config-file off --export ./from_cluster --force-overwrite --set full --import-source yes --source-folder ./benchmarks ./build/bench/cuda/bins/2.2_gpu_scattered_textureO3 ./data/mawi_201512020330/mawi_201512020330.mtx