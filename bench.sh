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
module load cuda
make all
srun ./build/bench/cuda/bins/gpu_with_inputO0 ./data/mawi_201512020000/mawi_201512020000.mtx
srun ./build/bench/cuda/bins/gpu_with_inputO1 ./data/mawi_201512020000/mawi_201512020000.mtx
srun ./build/bench/cuda/bins/gpu_with_inputO2 ./data/mawi_201512020000/mawi_201512020000.mtx
srun ./build/bench/cuda/bins/gpu_with_inputO3 ./data/mawi_201512020000/mawi_201512020000.mtx

srun ./build/bench/std/bins/baselineO0
srun ./build/bench/std/bins/baselineO1
srun ./build/bench/std/bins/baselineO2
srun ./build/bench/std/bins/baselineO3