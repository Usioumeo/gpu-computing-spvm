// USELESS DEFINE, MAKES HAPPY THE LINTER

#include "lib.h"
#include <omp.h>
#include <stdio.h>
#include <sys/time.h>
#include <immintrin.h>



int spmv_csr_simd_ilp_openmp(CSR csr, unsigned n, float *restrict input_vec,
                         float *restrict output_vec) {
                         
  if (n != csr.ncol) {
    return 1;
  }
  #pragma omp parallel for schedule(static, 64)
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
  CSR *csr = read_from_file(argc, argv);
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


  csr_free(csr);
  free(rand_vec);
  free(output);
  printf("test passed\n");
  return 0;
}