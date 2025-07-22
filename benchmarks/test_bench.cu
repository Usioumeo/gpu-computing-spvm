extern "C" {
#include "lib.h"
}
#include <math.h>
#include <stdio.h>
#include <sys/select.h>
#include <sys/time.h>
#define ROWS 35991342
//(1 << 25)
#define COLS 35991342
//(1 << 25)
#define NNZ 37242710
//(1 << 26)

#define BLOCK_SIZE 64

__global__ void spmv_csr_gpu_kernel(CSR csr, unsigned n, float *input_vec,
                                    float *output_vec) {
  // for (unsigned i = 0; i < csr.nrow; ++i) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < csr.nrow) {
    float out = 0.0;
    unsigned start = csr.row_idx[i];
    unsigned end = csr.row_idx[i + 1];

    float *val = csr.val + start;
    unsigned *col = csr.col_idx + start;
    float *val_end = csr.val + end;

    while (val < val_end) {

      out += *val * input_vec[*col];
      val++;
      col++;
    }
    output_vec[i] = out;
  }

  //}
}
int spmv_csr_gpu(CSR csr, unsigned n, float *input_vec, float *output_vec) {
  if (n != csr.ncol) {
    return 1;
  }

  unsigned int nblocks = (csr.nrow + BLOCK_SIZE - 1) / BLOCK_SIZE;
  printf("nblocks %u\n", nblocks);
  spmv_csr_gpu_kernel<<<nblocks, BLOCK_SIZE>>>(csr, n, input_vec, output_vec);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  return 0;
}

int main(void) {
  COO *coo = coo_new();
  // cudaMallocManaged(&coo, sizeof(COO));
  coo_generate_random(coo, ROWS, COLS, NNZ);
  CSR *csr = csr_new();
  coo_to_csr(coo, csr);

  float *rand_vec; // = (float * )malloc(sizeof(float)*COLS);
  cudaMallocManaged(&rand_vec, COLS * sizeof(float));
  float *output; //= (float*)malloc(sizeof(float)*COLS*(REPS+1));
  cudaMallocManaged(&output, COLS * 2 * sizeof(float));
  memset(output, 0, sizeof(float) * COLS * 2);
  for (unsigned i = 0; i < COLS; i++) {
    rand_vec[i] = (float)(rand() % 2001 - 1000) * 0.001;
  }

  printf("coo->nrow %u coo->ncol %u coo->nnz %u\n", coo->nrow, coo->ncol,
         coo->nnz);
  TEST_FUNCTION(spmv_csr_gpu(*csr, COLS, rand_vec, output));

  spmv_csr(*csr, COLS, rand_vec, &output[COLS]);

  if (relative_error_compare(output, output + csr->ncol, csr->ncol)) {
    printf("Error in the output\n");
    return -1;
  }

  coo_free(coo);
  csr_free(csr);
  cudaFree(rand_vec);
  cudaFree(output);
  // free(rand_vec);
  // free(output);
  // printf("test passed\n");
  return 0;
}
