extern "C" {
#include"lib.h"
}
#include <math.h>
#include <stdio.h>
#include <sys/select.h>
#include <sys/time.h>
#define ROWS (1 << 13)
#define COLS (1 << 13)
#define NNZ (1 << 24)

#define WARMUPS 40
#define REPS 10

#define BLOCK_SIZE 64




__global__ void spmv_csr_gpu_kernel(CSR csr, unsigned n, float *input_vec,
                                    float *output_vec) {
  // for (unsigned i = 0; i < csr.nrow; ++i) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < csr.nrow) {
    float out = 0.0;
    unsigned j = csr.row_idx[i];
    unsigned end = csr.row_idx[i + 1];
    unsigned *col_idx = csr.col_idx;
    float *val = csr.val;

    for (; j < end; ++j) {
      out += val[j] * input_vec[col_idx[j]];
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

  /*for (unsigned i = 0; i < csr.nrow; ++i) {
    output_vec[i] = 0.0;
    for (unsigned j = csr.row_idx[i]; j < csr.row_idx[i + 1]; ++j) {
      output_vec[i] += csr.val[j] * input_vec[csr.col_idx[j]];
    }
  }*/

  return 0;
}
int main(int argc, char *argv[]) {
    if(argc != 2) {
        printf("Usage: %s <input_file>\n", argv[0]);
        return -1;
    }

    FILE *input = fopen(argv[1], "r");
    if (input == NULL) {
        printf("Error opening file: %s\n", argv[1]);
        return -1;
    }

  COO *coo = coo_new();
  if (coo_from_file(input, coo)){
    printf("Error reading COO from file: %s\n", argv[1]);
    fclose(input);
    return -1;
  }
  CSR *csr = csr_new();
  printf("coo->nnz %u\n", coo->nnz);
  coo_to_csr(coo, csr);
  printf("converted\n");

  float *rand_vec; // = (float * )malloc(sizeof(float)*COLS);
  cudaMallocManaged(&rand_vec, csr->ncol * sizeof(float));
  float *output; //= (float*)malloc(sizeof(float)*COLS*(REPS+1));
  cudaMallocManaged(&output, csr->ncol * 2 * sizeof(float));
  memset(output, 0, sizeof(float) * csr->ncol * 2);
  for (unsigned i = 0; i < csr->ncol; i++) {
    rand_vec[i] = (float)(rand() % 2001 - 1000) * 0.001;
  }
  printf("starting the test\n");

  START_TIMER
  spmv_csr_gpu(*csr, csr->ncol, rand_vec, output);
  END_TIMER

  spmv_csr(*csr, csr->ncol, rand_vec, &output[COLS]);
  printf("Elapsed time: %f\n in order to do %u (avaraged on reps %u)\n",
         CPU_time, REPS, csr->nnz);

  float flops = 2.0 * NNZ / CPU_time;
  printf("computed Gflops = %f\n", flops / 1.0e9);
  
  size_t total_memory = (csr->nrow) * sizeof(unsigned) * 2 +
                        csr->nnz * (sizeof(float) + sizeof(unsigned)) +
                        csr->nnz * sizeof(float) + csr->nrow * sizeof(unsigned);
  float gbytes = (float)total_memory / 1.0e9;
  float gbytesps = gbytes / CPU_time;
  printf("total memory = %f GB\n", gbytes);
  printf("total memory = %f GB/s\n", gbytesps);

  // printf("output %lu\n", out-output);
  for (unsigned j = 0; j < csr->ncol; j++) {

    if (output[j] - output[csr->ncol + j] > 0.001) {
      printf("Error in the output %u %f %f %u %u\n", j, output[j],
             output[csr->ncol + j], j, csr->ncol + j);
      return -1;
    }
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
