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
make all
#ARGUMENT = ./data/mawi_201512020000/mawi_201512020000.mtx
#ARGUMENT = ./data/mawi_201512020330/mawi_201512020330.mtx
RUNNER=
#RUNNER= srun
ARGUMENT= 
$RUNNER ./build/bench/std/bins/1_baseline_single_coreO3 $ARGUMENT
$RUNNER ./build/bench/std/bins/2_simd_ilpO3 $ARGUMENT
$RUNNER ./build/bench/std/bins/3_simd_ilp_openmpO3 $ARGUMENT
$RUNNER ./build/bench/std/bins/4_simd_ilp_openmp_blockO3 $ARGUMENT
$RUNNER ./build/bench/std/bins/5_simd_ilp_openmp_nnz_blockO3 $ARGUMENT

$RUNNER ./build/bench/cuda/bins/1_gpu_baselineO3 $ARGUMENT
$RUNNER ./build/bench/cuda/bins/2_gpu_scatteredO3 $ARGUMENT
$RUNNER ./build/bench/cuda/bins/0_cusparseO3 $ARGUMENT
