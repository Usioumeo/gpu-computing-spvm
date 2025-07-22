// USELESS DEFINE, MAKES HAPPY THE LINTER
#define USE_OPENMP
#include "lib.h"
#include <omp.h>
#include <stdio.h>
#include <sys/time.h>






int main(int argc, char *argv[]) {
  printf("baseline single core\n");
  CSR *csr = common_read_from_file(argc, argv);

  float *input = common_generate_random_input(csr);
  float *output = (float *)malloc(sizeof(float) *  csr->nrow * 2);

  TEST_FUNCTION(spmv_csr(*csr,  csr->ncol, input, output);)

  spmv_csr(*csr,  csr->ncol, input, output +  csr->nrow);

  if (relative_error_compare(output, output + csr->nrow, csr->nrow)) {
    printf("Error in the output\n");
    return -1;
  }

  csr_free(csr);
  free(input);
  free(output);
  return 0;
}