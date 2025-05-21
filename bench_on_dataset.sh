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
module load CUDA/12.1
echo default
$EXEC
echo mawi1
$EXEC ./data/mawi_201512020000/mawi_201512020000.mtx
echo random2
$EXEC ./data/random2.mtx
echo mawi2
$EXEC ./data/mawi_201512020330/mawi_201512020330.mtx
echo nlppkkt240
$EXEC ./data/nlpkkt240/nlpkkt240.mtx
echo