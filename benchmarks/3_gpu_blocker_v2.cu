#include <cassert>
#include <cstdlib>
extern "C" {
#include "lib.h"
}
#include <math.h>
#include <stdio.h>
#include <sys/select.h>
#include <sys/time.h>


#define BLOCK_SIZE (32 * 2)
#define SHARED_DATA_BLOCK (1024)
__inline__ __device__ unsigned warpReduceMin(unsigned val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val = min(val, __shfl_down_sync(0xffffffff, val, offset));
  }

  return __shfl_sync(0xffffffff, val, 0);
  // return val;
}

__inline__ __device__ unsigned blockReduceMin(unsigned val) {
  __shared__ unsigned partialMins[BLOCK_SIZE / 32];
  unsigned lane_id = threadIdx.x % 32;
  unsigned warp_id = threadIdx.x / 32;

  for (int offset = 16; offset > 0; offset /= 2) {
    val = min(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  if (lane_id == 0) {
    partialMins[warp_id] = val;
  }

  __syncthreads();
  // accumulate partial accumulation
  if (threadIdx.x < 32) {
    val = threadIdx.x < BLOCK_SIZE / 32 ? partialMins[threadIdx.x] : 0xffffffff;
    // Reduce within this warp to find global minimum
    for (int offset = 16; offset > 0; offset /= 2) {
      val = min(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    val = __shfl_sync(0xffffffff, val, 0);
    partialMins[threadIdx.x] = val;
  }
  __syncthreads();
  if (lane_id == 0) {
    val = partialMins[warp_id];
  }
  __syncwarp();
  return __shfl_sync(0xffffffff, val, 0);
}

__inline__ __device__ unsigned warpReduceMax(unsigned val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val = max(val, __shfl_down_sync(0xffffffff, val, offset));
  }

  return __shfl_sync(0xffffffff, val, 0);
  // return val;
}

__global__ void spmv_csr_gpu_kernel_blocks(CSR csr, unsigned n,
                                           float *__restrict__ input_vec,
                                           float *output_vec) {
  __shared__ float shared_input[SHARED_DATA_BLOCK];

  // for (unsigned i = 0; i < csr.nrow; ++i) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  float out = 0.0;
  unsigned pos;
  unsigned end;
  if (i < csr.nrow) {
    pos = csr.row_idx[i];
    end = csr.row_idx[i + 1];
  } else {
    end = csr.nnz; // If i >= nrow, set end to nnz to avoid out-of-bounds access
    pos = csr.nnz; // If i >= nrow, set pos to nnz to avoid out-of-bounds access
  }

  __syncthreads();
  unsigned end_block = 0;
  
  while (end_block < csr.ncol) {
    unsigned to_send =
        (pos < end && i < csr.nrow) ? csr.col_idx[pos] : csr.ncol;

    //__syncwarp();
    // assert(__activemask() ==0xffffffff);
    __syncthreads();
    unsigned min_col = blockReduceMin(to_send);
    // printf("min_col %u\n", min_col);
    end_block = min(min_col + SHARED_DATA_BLOCK, csr.ncol);
    __syncthreads();
    for (unsigned j = min_col + threadIdx.x; j < end_block; j += BLOCK_SIZE) {

      shared_input[j - min_col] = __ldg(input_vec + j);
    }
    __syncthreads();

    /*unsigned col_val=*col;
    while (pos < end && col_val < end_block) {
      out += *val * shared_input[col_val - min_col];
      col_val=*++col;
      pos++;
      //col++;
      val++;

    }*/
    unsigned col_val = (pos < end) ? csr.col_idx[pos] : end_block;
    float val_current = (pos < end) ? csr.val[pos] : 0.0f;

    while (pos < end && col_val < end_block) {
      // Use current values
      out += val_current * shared_input[col_val - min_col];

      // Prefetch next iteration
      pos++;
      col_val = (pos < end) ? csr.col_idx[pos] : end_block;
      val_current = (pos < end) ? csr.val[pos] : 0.0f;
    }
  }
  output_vec[i] = out;

  //}
}

void dummy_launcher(CSR *csr, float *input_vec, float *output_vec) {
  unsigned nblocks = (csr->nrow + BLOCK_SIZE - 1) / BLOCK_SIZE;
  spmv_csr_gpu_kernel_blocks<<<nblocks, BLOCK_SIZE>>>(*csr, csr->ncol,
                                                      input_vec, output_vec);
  CHECK_CUDA(cudaDeviceSynchronize());
}
int spmv_csr_gpu_chunks(CSR *csr, unsigned n, float *input_vec,
                        float *output_vec) {
  if (n != csr->ncol) {
    return 1;
  }
  CSR *gpu_csr = copy_csr_to_gpu(csr);

  float *input_vec_gpu, *output_gpu;
  CHECK_CUDA(cudaMalloc(&input_vec_gpu, sizeof(float) * csr->ncol));
  CHECK_CUDA(cudaMemcpy(input_vec_gpu, input_vec, sizeof(float) * csr->ncol,
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&output_gpu, sizeof(float) * gpu_csr->nrow));

  TEST_FUNCTION(dummy_launcher(gpu_csr, input_vec_gpu, output_gpu));

  CHECK_CUDA(cudaMemcpy(output_vec, output_gpu, sizeof(float) * gpu_csr->nrow,
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(input_vec_gpu));
  CHECK_CUDA(cudaFree(output_gpu));
  free_csr_gpu(gpu_csr);
  return 0;
}

int main(int argc, char *argv[]) {
  CSR *csr = common_read_from_file(argc, argv);

   

  float *input = common_generate_random_input(csr);

  float *output = (float *)malloc(sizeof(float) * csr->nrow * 2);

  spmv_csr_gpu_chunks(csr, csr->ncol, input, output); //, tmp
  spmv_csr(*csr, csr->ncol, input, output + csr->nrow);

  if (relative_error_compare(output, output + csr->nrow, csr->nrow)) {
    printf("Error in the output\n");
    return 0;
  }

  csr_free(csr);
  free(input);
  free(output);
  printf("test passed\n\n");
  return 0;
}
