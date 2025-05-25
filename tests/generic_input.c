
#include"lib.h"
#include <stdio.h>
#include <string.h>
#include <sys/select.h>
#include <sys/time.h>
#define ROWS (1 << 13)
#define COL (1 << 13)
#define NNZ (1 << 24)

#define WARMUPS 40
#define REPS 10000

#define BLOCK_SIZE 64



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
  if (coo_from_file(input, coo)!=0){
    printf("Error reading COO from file: %s\n", argv[1]);
    fclose(input);
    return -1;
  }
  for(int i = 0; i < coo->nnz; i++) {
    if(coo->data[i].row >= coo->nrow || coo->data[i].col >= coo->ncol) {
      printf("Error: COO entry out of bounds (%u, %u, %u, %u, %u)\n", i, coo->nnz, coo->nrow, coo->data[i].row, coo->data[i].col);
      fclose(input);
      return -1;
    }

  }
  CSR *csr = csr_new();
  printf("loaded COO %u %u %u\n", coo->nrow, coo->ncol, coo->nnz);
  coo_to_csr(coo, csr);
  printf("converted to CSR %u %u %u\n", csr->nrow, csr->ncol, csr->nnz);
  float *rand_vec = (float * )malloc(sizeof(float)*csr->ncol);
  float *output= (float*)malloc(sizeof(float)*csr->ncol*2);
  memset(output, 0, sizeof(float) * csr->ncol * 2);
  for (unsigned i = 0; i < csr->ncol; i++) {
    rand_vec[i] = (float)(rand() % 2001 - 1000) * 0.001;
  }
  printf("vector %u %f\n", csr->ncol, rand_vec[0]);

  START_TIMER
  spmv_csr_openmp_simd(*csr, csr->ncol, rand_vec, output);
  END_TIMER

  spmv_csr(*csr, csr->ncol, rand_vec, &output[csr->ncol]);
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
  free(rand_vec);
  free(output);
  // free(rand_vec);
  // free(output);
  // printf("test passed\n");
  return 0;
}
