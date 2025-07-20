#include <cassert>
#include <cstdlib>
extern "C" {
#include "lib.h"
}
#include <math.h>
#include <stdio.h>
#include <sys/select.h>
#include <sys/time.h>
#define ROWS (1 << 13)
#define COLS (1 << 13)
#define NNZ (1 << 24)

#define WARMUPS 0
#define REPS 2

#define BLOCK_SIZE 128
#define SHARED_DATA_BLOCK 2048
__inline__ __device__ unsigned warpReduceMin(unsigned val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val = min(val, __shfl_down_sync(0xffffffff, val, offset));
  }

  return __shfl_sync(0xffffffff, val, 0);
  //return val;
}
__global__ void spmv_csr_gpu_kernel_blocks(CSR csr, unsigned n,
                                           float *__restrict__ input_vec,
                                           float *output_vec) {
  __shared__ float shared_input[SHARED_DATA_BLOCK*BLOCK_SIZE/32];

  // for (unsigned i = 0; i < csr.nrow; ++i) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned warp_id = threadIdx.x / 32;
  unsigned lane_id = threadIdx.x % 32;
  unsigned shared_offset = warp_id * SHARED_DATA_BLOCK;
  float out = 0.0;
  unsigned pos;
  unsigned end;
  if(i<csr.nrow){
    pos = csr.row_idx[i];
    end = csr.row_idx[i + 1];
  } else {
    end = csr.nnz; // If i >= nrow, set end to nnz to avoid out-of-bounds access
    pos = csr.nnz; // If i >= nrow, set pos to nnz to avoid out-of-bounds access
  }
  
  
  const float *__restrict__ val = csr.val + pos;
  /*const unsigned *__restrict__ col = csr.col_idx + start;
  const float *__restrict__ val_end = csr.val + end;*/
  __syncthreads();
  unsigned end_block = 0;
  while (end_block< csr.ncol) {
    unsigned to_send =
        (pos < end && i < csr.nrow) ? csr.col_idx[pos] : csr.ncol;

    __syncwarp();
    assert(__activemask() ==0xffffffff);
    unsigned min_col = warpReduceMin(to_send);
    end_block = min(min_col + SHARED_DATA_BLOCK, csr.ncol);
    __syncwarp();
    for(unsigned i=min_col+lane_id; i < end_block; i += 32) {
      shared_input[i-min_col+shared_offset] = __ldg(input_vec + i);
    }
    __syncwarp();




    while (pos < end && csr.col_idx[pos] < end_block) {
      out += *val* shared_input[csr.col_idx[pos]-min_col+shared_offset];
      pos++;
      val++;
    }
    output_vec[i] = out;
  }

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
  COO *coo = coo_new();
  CSR *csr = csr_new();
  if (argc > 2) {
    printf("Usage: %s <input_file>\n", argv[0]);
    return -1;
  }
  if (argc == 2) {
    FILE *input = fopen(argv[1], "r");
    if (input == NULL) {
      printf("Error opening file: %s\n", argv[1]);
      return -1;
    }
    if (coo_from_file(input, coo) != 0) {
      printf("Error reading COO from file: %s\n", argv[1]);
      fclose(input);
      return -1;
    }
    coo_to_csr(coo, csr);
    write_bin_to_file(csr, "tmp.bin");
  } else {
    // coo_generate_random(coo, ROWS, COLS, NNZ);
    read_bin_to_csr("tmp.bin", csr);
  }

  printf("csr->nrow %u csr->ncol %u csr->nnz %u\n", csr->nrow, csr->ncol,
         csr->nnz);

  float *input = (float *)malloc(sizeof(float) * csr->ncol);
  // cudaMallocHost(&rand_vec_host, sizeof(float)*COLS);
  for (unsigned i = 0; i < csr->ncol; i++) {
    input[i] = (float)(rand() % 2001 - 1000) * 0.001;
  }

  float *output = (float *)malloc(sizeof(float) * csr->nrow * 2);

  spmv_csr_gpu_chunks(csr, csr->ncol, input, output); //, tmp
  spmv_csr(*csr, csr->ncol, input, output + csr->nrow);

  if (relative_error_compare(output, output + csr->nrow, csr->nrow)) {
    printf("Error in the output\n");
    return 0;
  }

  coo_free(coo);
  csr_free(csr);
  free(input);
  free(output);
  printf("test passed\n\n");
  return 0;
}
