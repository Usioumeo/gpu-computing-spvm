
#include <cassert>
extern "C" {
#include "lib.h"
}
#include <math.h>
#include <stdio.h>
#include <sys/select.h>
#include <sys/time.h>

#define BLOCK_SIZE 32
#define DATA_BLOCK (16)

__device__ unsigned upper_bound(const unsigned *arr, int size, unsigned key) {
  int left = 0;
  int right = size;
  while (left + 1 < right) {
    // printf("left %u right %u\n", left, right);
    int mid = (right + left) / 2;
    if (arr[mid] <= key)
      left = mid;
    else
      right = mid;
  }
  return left;
}

__global__ void spmv_csr_gpu_kernel_chunks(const float *__restrict__ val,
                                           const unsigned *__restrict__ row_idx,
                                           const unsigned *__restrict__ col_idx,
                                           const float *__restrict__ input_vec,
                                           float *output_vec, unsigned nrow,
                                           unsigned ncol, unsigned nnz) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned starting_point = i * DATA_BLOCK;
  if (starting_point >= nnz) {
    return;
  }
  unsigned row = upper_bound(row_idx, nrow, starting_point);
  if (row_idx[row] > starting_point || starting_point >= row_idx[row + 1]) {

    printf("Hello from thread %d, block %d\n", threadIdx.x, blockIdx.x);
    printf("row %u %u <= %u < %u\n", row, row_idx[row], starting_point,
           row_idx[row + 1]);
    assert(1 == 0);
  }
  unsigned next_row_pos = row_idx[row + 1];
  float cur_val = 0.0;
  unsigned end_point = starting_point + DATA_BLOCK;
  if (end_point > nnz) {
    end_point = nnz;
  }
  const unsigned *next_row_idx = &row_idx[row + 1];
  for (unsigned j = starting_point; j < end_point; j++) {
    if (j == next_row_pos) {
      // printf("row %u += %f\n", row, cur_val);
      // exit(1);
      //  THIS MUST BE DONE IN A ATOMIC WAY
      // output_vec[row]+=cur_val;
      atomicAdd(&output_vec[row], cur_val);
      cur_val = 0.0;
      while (j == next_row_pos) {
        ++row;
        next_row_pos = *(++next_row_idx);
      }
    }
    cur_val += val[j] * input_vec[col_idx[j]];
  }

  // output_vec[row] += cur_val;
  atomicAdd(&output_vec[row], cur_val);
}

int spmv_csr_gpu_chunks(CSR csr, unsigned n, float *input_vec,
                        float *output_vec) {
  if (n != csr.ncol) {
    return 1;
  }
  /*for (int i = 0; i < csr.nrow; i++) {
    output_vec[i] = 0.0;
  }*/
  cudaMemset(output_vec, 0, sizeof(float) * csr.nrow);
  unsigned n_data_blocks = (csr.nnz + DATA_BLOCK - 1) / DATA_BLOCK;
  unsigned nblocks = (n_data_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
  /*printf("nblocks %u\n", nblocks);
  printf("n_data_blocks %u\n", n_data_blocks);*/
  spmv_csr_gpu_kernel_chunks<<<nblocks, BLOCK_SIZE>>>(
      csr.val, csr.row_idx, csr.col_idx, input_vec, output_vec, csr.nrow,
      csr.ncol, csr.nnz);
  cudaDeviceSynchronize();

  return 0;
}

__global__ void spmv_csr_gpu_kernel_nnz(const float *__restrict__ val,
                                        const unsigned *__restrict__ row_idx,
                                        const unsigned *__restrict__ col_idx,
                                        const float *__restrict__ input_vec,
                                        float *output_vec, unsigned nrow,
                                        unsigned ncol, unsigned nnz) {
  // for (unsigned i = 0; i < csr.nrow; ++i) {
  unsigned start = (blockIdx.x * blockDim.x + threadIdx.x) * DATA_BLOCK;

  unsigned end = start + DATA_BLOCK;
  if (start >= nnz) {
    return;
  }
  if (end > nnz) {
    end = nnz;
  }
  float cur = 0.0;
  unsigned prev_row = upper_bound(row_idx, nrow, start);
  for (int i = start; i < end; i++) {
    unsigned cur_row = upper_bound(row_idx, nrow, i);
    if (prev_row != cur_row) {
      atomicAdd(&output_vec[prev_row], cur);
      cur = 0.0;
      prev_row = cur_row;
    }
    cur += val[i] * input_vec[col_idx[i]];
  }
  atomicAdd(&output_vec[prev_row], cur);
}

int spmv_csr_gpu_nnz(CSR csr, unsigned n, float *input_vec, float *output_vec,
                     unsigned *rows) {
  if (n != csr.ncol) {
    return 1;
  }
  cudaMemset(output_vec, 0, sizeof(float) * csr.nrow);
  unsigned int nblocks = (csr.nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;
  /*set_rows<<<nblocks, BLOCK_SIZE>>>(csr, rows);
  cudaDeviceSynchronize();*/
  unsigned n_data_blocks = (csr.nnz + DATA_BLOCK - 1) / DATA_BLOCK;
  nblocks = (n_data_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
  spmv_csr_gpu_kernel_nnz<<<nblocks, BLOCK_SIZE>>>(
      csr.val, csr.row_idx, csr.col_idx, input_vec, output_vec, csr.nrow,
      csr.ncol, csr.nnz);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  return 0;
}

int main(int argc, char *argv[]) {
  CSR *csr = common_read_from_file(argc, argv);

  float *rand_vec; // = (float * )malloc(sizeof(float)*COLS);
  cudaMallocManaged(&rand_vec, csr->ncol * sizeof(float));
  float *output; //= (float*)malloc(sizeof(float)*COLS*(REPS+1));
  cudaMallocManaged(&output, sizeof(float) * csr->nrow * 2);
  for (unsigned i = 0; i < csr->ncol; i++) {
    rand_vec[i] = (float)(rand() % 2001 - 1000) * 0.001;
  }
  unsigned *tmp;
  cudaMallocManaged(&tmp, csr->nnz * sizeof(unsigned));
  TEST_FUNCTION(spmv_csr_gpu_chunks(*csr, csr->ncol, rand_vec, output)); //, tmp
  spmv_csr(*csr, csr->ncol, rand_vec, output + csr->nrow);

  if (relative_error_compare(output, output + csr->nrow, csr->nrow)) {
    printf("Error in the output\n");
    return -1;
  }

  csr_free(csr);
  cudaFree(rand_vec);
  cudaFree(output);
  cudaFree(tmp);
  printf("test passed\n\n");
  // free(rand_vec);
  // free(output);
  // printf("test passed\n");
  return 0;
}
