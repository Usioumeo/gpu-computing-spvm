#include <cassert>
extern "C" {
#define USE_CUDA
#include "lib.h"
}
#include <math.h>
#include <stdio.h>
#include <sys/select.h>
#include <sys/time.h>

#include <stdint.h>  
#define ROWS (1 << 13)
#define COLS (1 << 13)
#define NNZ (1 << 24)

#define WARMUPS 4
#define REPS 10

#define BLOCK_SIZE 16
#define ROW_PER_BLOCK 16
__global__ void spmv_csr_gpu_kernel(CSR csr, unsigned n, float *__restrict__ input_vec,
                                    float *output_vec) {
  __shared__ float shared_output[ROW_PER_BLOCK];
  for(unsigned i=threadIdx.x; i<ROW_PER_BLOCK; i+=BLOCK_SIZE){
    shared_output[i] = 0.0;
  }
  
  //unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  __syncthreads();
  unsigned starting_row = blockIdx.x*ROW_PER_BLOCK;
  unsigned end_row = (blockIdx.x+1)*ROW_PER_BLOCK<csr.nrow ? (blockIdx.x+1)*ROW_PER_BLOCK : csr.nrow;
  for (unsigned i = starting_row; i<end_row; i++) {
    float out = 0.0;
    unsigned start = csr.row_idx[i]+threadIdx.x;
    unsigned end = csr.row_idx[i + 1];

    float *val = csr.val + start;
    unsigned *col = csr.col_idx + start;
    float *val_end = csr.val + end;
    unsigned col_val=__ldg(col);
    while (val < val_end) {
      out += *val * __ldg(&input_vec[col_val]);
      
      col+= BLOCK_SIZE;
      col_val=__ldg(col);
      val+= BLOCK_SIZE;
      
    }
    atomicAdd(&shared_output[i-starting_row], out);
    //__syncthreads();
    //shared_output[i] += out;
    //output_vec[i] = out;
  }
  __syncthreads();
  for(unsigned i=threadIdx.x+starting_row; i<end_row; i+=BLOCK_SIZE){
    output_vec[i]=shared_output[i-starting_row];
  }
  /*// for (unsigned i = 0; i < csr.nrow; ++i) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < csr.nrow) {
    float out = 0.0;
    unsigned start = csr.row_idx[i];
    unsigned end = csr.row_idx[i + 1];

    float *val = csr.val + start;
    unsigned *col = csr.col_idx + start;
    float *val_end = csr.val + end;

    while (val < val_end) {
      
      out += *val * __ldg(&input_vec[*col]);
      val++;
      col++;
    }
    output_vec[i] = out;
  }

  //}*/
}

void dummy_launcher(CSR *csr, float *input_vec, float *output_vec) {
  unsigned nblocks = (csr->nrow + ROW_PER_BLOCK- 1) / ROW_PER_BLOCK;
  spmv_csr_gpu_kernel<<<nblocks, BLOCK_SIZE>>>(*csr, csr->ncol, input_vec,
                                               output_vec);
  CHECK_CUDA(cudaDeviceSynchronize());
}

int spmv_csr_gpu(CSR *csr, unsigned n, float *input_vec,
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
    //coo_generate_random(coo, ROWS, COLS, NNZ);
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

  spmv_csr_gpu(csr, csr->ncol, input, output); //, tmp
  spmv_csr(*csr, csr->ncol, input, output + csr->nrow);

  if (relative_error_compare(output, output + csr->nrow, csr->nrow)) {
    printf("Error in the output\n");
    return -1;
  }

  coo_free(coo);
  csr_free(csr);
  free(input);
  free(output);
  printf("test passed\n\n");
  return 0;
}