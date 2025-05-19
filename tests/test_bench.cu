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
#define REPS 10000

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

// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y) {
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
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
  // CSR *temp=csr_new();
  //  csr_sort_in_ascending_order(*csr);

  START_TIMER
  spmv_csr_gpu(*csr, COLS, rand_vec, output);
  END_TIMER

  spmv_csr(*csr, COLS, rand_vec, &output[COLS]);
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
  for (unsigned j = 0; j < COLS; j++) {

    if (output[j] - output[COLS + j] > 0.001) {
      printf("Error in the output %u %f %f %u %u\n", j, output[j],
             output[COLS + j], j, COLS + j);
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
