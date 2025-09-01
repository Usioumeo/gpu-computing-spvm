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
#make all
#ARGUMENT=./data/mawi_201512020000/mawi_201512020000.mtx
ARGUMENT=$1
RUNNER=
#RUNNER=srun
#ARGUMENT= 
#$RUNNER ./build/bench/std/bins/1_baseline_single_coreO3 $ARGUMENT
#$RUNNER ./build/bench/std/bins/2_simd_ilpO3 $ARGUMENT
#$RUNNER ./build/bench/std/bins/3_simd_ilp_openmpO3 $ARGUMENT
#$RUNNER ./build/bench/std/bins/4_openmp_nnz_blocksO3 $ARGUMENT
#$RUNNER ./build/bench/std/bins/5_simd_ilp_openmp_blockO3 $ARGUMENT


echo "1 baseline"
./build/bench/cuda/bins/1_gpu_baselineO3 $ARGUMENT
echo "1 texture"
./build/bench/cuda/bins/1_gpu_baseline_textureO3 
echo "2 row"
./build/bench/cuda/bins/2.2_gpu_scattered_v2_rowO3
echo "2v2"
./build/bench/cuda/bins/2.2_gpu_scattered_v2O3
echo "2 texture"
./build/bench/cuda/bins/2.2_gpu_scattered_textureO3
echo "3 v3"
./build/bench/cuda/bins/3_gpu_blocker_v3O3
echo "4 mix"
./build/bench/cuda/bins/2.4_gpu_mixedO3
echo "cusparse"
#$RUNNER ./build/bench/cuda/bins/2_gpu_scatteredO3 $ARGUMENT
./build/bench/cuda/bins/0_cusparse_alg1O3
