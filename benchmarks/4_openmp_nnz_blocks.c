// USELESS DEFINE, MAKES HAPPY THE LINTER

#include "lib.h"
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#define ROWS (1 << 13)
#define COLS (1 << 13)
#define NNZ (1 << 24)

#define WARMUPS 0
#define REPS 1

#define BLOCK_SIZE (1<<12)
inline static float simd_process(CSR csr, unsigned start, unsigned end, float *input_vec
                                 ) {
  float ret = 0.0;
  unsigned aligned_start = (start + 7) & ~7;
  unsigned aligned_end = end & ~7;
  if (aligned_start > aligned_end) {
    for (unsigned k = start; k < end; k++) {
      ret += csr.val[k] * input_vec[csr.col_idx[k]];
    }
    return ret;
  }

  for (unsigned k = start; k < aligned_start; k++) {
    ret += csr.val[k] * input_vec[csr.col_idx[k]];
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
    ret += temp[j];
  }

  for (unsigned j = aligned_end; j < end; ++j) {
    ret += csr.val[j] * input_vec[csr.col_idx[j]];
  }
  return ret;
}
unsigned upper_bound(const unsigned *arr, int size, unsigned key) {
  int left = 0;
  int right = size;
  while (left + 1 < right) {
    // printf("left %u right %u\n", left, right);
    int mid = (right + left) / 2;
    if (arr[mid] <= key)
      left = mid;
    else
      right = mid;
  }
  return left;
}
int spmv_csr_simd_ilp_openmp(CSR csr, unsigned n, float *input_vec,
                             float *output_vec) {

  if (n != csr.ncol) {
    return 1;
  }
  for(unsigned i=0; i<csr.nrow; i++){
    output_vec[i] = 0.0;
  }

  unsigned blocks = (csr.nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;
#pragma omp parallel for schedule(dynamic)
  for (unsigned b = 0; b < blocks; b++) {
    unsigned block_start = b * BLOCK_SIZE;
    unsigned block_max = (block_start + BLOCK_SIZE < csr.nnz)
                             ? (block_start + BLOCK_SIZE)
                             : csr.nnz;
    unsigned row = upper_bound(csr.row_idx, csr.nrow, block_start);
    do{
      unsigned start_row = (csr.row_idx[row]<block_start)?(block_start):(csr.row_idx[row]);
      unsigned end_row = (csr.row_idx[row + 1]<block_max)?(csr.row_idx[row + 1]):block_max;
      float cur= simd_process(csr, start_row, end_row, input_vec);
      #pragma omp critical
      {
        output_vec[row] += cur;
      }
      ++row;

    }while (row < csr.nrow && csr.row_idx[row] < block_max);
  }
  return 0;
}

int main(int argc, char *argv[]) {
  printf("simd ilp openmp nnz blocks\n");
  COO *coo = coo_new();
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
  } else {
    coo_generate_random(coo, ROWS, COLS, NNZ);
  }
  CSR *csr = csr_new();
  coo_to_csr(coo, csr);

  float *rand_vec = (float *)malloc(sizeof(float) * csr->ncol);
  float *output = (float *)malloc(sizeof(float) *  csr->nrow * 2);
  for (unsigned i = 0; i < csr->ncol; i++) {
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
  printf("test passed\n\n");
  return 0;
}