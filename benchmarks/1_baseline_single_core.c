// USELESS DEFINE, MAKES HAPPY THE LINTER
#define USE_OPENMP
#include "lib.h"
#include <omp.h>
#include <stdio.h>
#include <sys/time.h>






int main(int argc, char *argv[]) {
  printf("baseline single core\n");
  CSR *csr = read_from_file(argc, argv);

  float *rand_vec = (float *)malloc(sizeof(float) * csr->ncol);
  float *output = (float *)malloc(sizeof(float) *  csr->nrow * 2);
  for (unsigned i = 0; i <  csr->ncol; i++) {
    rand_vec[i] = (float)(rand() % 2001 - 1000) * 0.001;
  }
  printf("CSR matrix: %d rows, %d cols, %d nnz\n", csr->nrow, csr->ncol, csr->nnz);
  TEST_FUNCTION(spmv_csr(*csr,  csr->ncol, rand_vec, output);)

  spmv_csr(*csr,  csr->ncol, rand_vec, output +  csr->nrow);

  if (relative_error_compare(output, output + csr->nrow, csr->nrow)) {
    printf("Error in the output\n");
    return -1;
  }

  csr_free(csr);
  free(rand_vec);
  free(output);
  printf("test passed\n\n");
  return 0;
}