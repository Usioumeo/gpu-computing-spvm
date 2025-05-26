// USELESS DEFINE, MAKES HAPPY THE LINTER

#include "lib.h"
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <immintrin.h>
#define ROWS (1 << 13)
#define COLS (1 << 13)
#define NNZ (1 << 24)

#define WARMUPS 40
#define REPS 100


int spmv_csr_simd_ilp_openmp(CSR csr, unsigned n, float *input_vec,
                         float *output_vec) {
                         
  if (n != csr.ncol) {
    return 1;
  }
  #pragma omp parallel for schedule(static, 1)
  for (unsigned i = 0; i < csr.nrow; ++i) {
    output_vec[i] = 0.0;
    unsigned start = csr.row_idx[i];
    unsigned aligned_start = (start + 7) & ~7;
    unsigned end = csr.row_idx[i + 1];
    unsigned aligned_end = end & ~7;
    if (aligned_start > aligned_end){
      for (unsigned k = start; k < end; k++) {
        output_vec[i] += csr.val[k] * input_vec[csr.col_idx[k]];
      }
      continue;
    }
      
    for (unsigned k = start; k < aligned_start; k++) {
      output_vec[i] += csr.val[k] * input_vec[csr.col_idx[k]];
    }
    __m256 cumulate = _mm256_setzero_ps();

    for (unsigned j = aligned_start; j < aligned_end; j += 8) {
      // load 8 offsets
      __m256 vec1 = _mm256_loadu_ps(&csr.val[j]);
      // Gather indices as 8 32-bit integers
      __m256i indices = _mm256_loadu_si256((const __m256i *)&csr.col_idx[j]);
      __m256 vec2 = _mm256_i32gather_ps(input_vec, indices, 4);
      __m256 product = _mm256_mul_ps(vec1, vec2);
      cumulate = _mm256_add_ps(cumulate, product);
    }

    float temp[8];
    _mm256_storeu_ps(temp, cumulate);
    for (int j = 0; j < 8; j++) {
      output_vec[i] += temp[j];
    }
    
    for (unsigned j = aligned_end; j < end; ++j) {
      output_vec[i] += csr.val[j] * input_vec[csr.col_idx[j]];
    }
  }
  return 0;
}


int main(int argc, char *argv[]) {
  printf("simd ilp openmp\n\n");
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
  float *rand_vec = (float *)malloc(sizeof(float) * csr->ncol);
  float *output = (float *)malloc(sizeof(float) *  csr->nrow * 2);
  for (unsigned i = 0; i <  csr->ncol; i++) {
    rand_vec[i] = (float)(rand() % 2001 - 1000) * 0.001;
  }

  TEST_FUNCTION(spmv_csr_simd_ilp_openmp(*csr, csr->ncol, rand_vec, output);)

  spmv_csr(*csr,  csr->ncol, rand_vec, output +  csr->nrow);

  if (relative_error_compare(output, output + csr->nrow, csr->nrow)) {
    printf("Error in the output\n");
    return -1;
  }


  coo_free(coo);
  csr_free(csr);
  free(rand_vec);
  free(output);
  printf("test passed\n");
  return 0;
}