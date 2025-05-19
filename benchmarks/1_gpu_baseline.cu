
#include <cassert>
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

#define WARMUPS 40
#define REPS 100

#define BLOCK_SIZE 32
#define DATA_BLOCK (16)

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
  spmv_csr_gpu_kernel<<<nblocks, BLOCK_SIZE>>>(csr, n, input_vec, output_vec);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  return 0;
}

int main(int argc, char *argv[]) {
  printf("baseline single core\n");
  COO *coo = coo_new();
  if(argc > 2) {
    printf("Usage: %s <input_file>\n", argv[0]);
    return -1;
  }
  if (argc==2) {
    FILE *input = fopen(argv[1], "r");
    if (input == NULL) {
      printf("Error opening file: %s\n", argv[1]);
      return -1;
    }
    if (coo_from_file(input, coo)!=0){
      printf("Error reading COO from file: %s\n", argv[1]);
      fclose(input);
      return -1;
    }
  } else{
    coo_generate_random(coo, ROWS, COLS, NNZ);
  }
  CSR *csr = csr_new();
  coo_to_csr(coo, csr);

  float *rand_vec; //= (float *)malloc(sizeof(float) * csr->ncol);
  cudaMallocManaged(&rand_vec, sizeof(float) * csr->ncol);
  float *output; //= (float *)malloc(sizeof(float) *  csr->ncol * 2);
  cudaMallocManaged(&output, sizeof(float) * csr->ncol * 2);
  for (unsigned i = 0; i <  csr->ncol; i++) {
    rand_vec[i] = (float)(rand() % 2001 - 1000) * 0.001;
  }


  TEST_FUNCTION(spmv_csr_gpu(*csr, csr->ncol, rand_vec, output));
  spmv_csr(*csr, csr->ncol, rand_vec, &output[csr->ncol]);
  // printf("output %lu\n", out-output);
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
