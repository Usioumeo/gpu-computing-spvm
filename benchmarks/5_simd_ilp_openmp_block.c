// USELESS DEFINE, MAKES HAPPY THE LINTER

#include "lib.h"
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>

#define BLOCK_SIZE 512
#define BLOCK_ADV 32
typedef struct {
  float cur;
  unsigned row;
  unsigned *col_index_cur;
  unsigned *col_index_end;
  unsigned *col_index_aligned_end;
  float *val_cur;

  // char align[24]; // padding to ensure alignment
} BlockData;

inline static void process_step(BlockData *blocks, float *input_vec) {
  blocks->cur += *blocks->val_cur++ * input_vec[*blocks->col_index_cur++];
}
int spmv_csr_simd_ilp_openmp(CSR csr, unsigned n, float *input_vec,
                             float *output_vec) {

  if (n != csr.ncol) {
    return 1;
  }
  unsigned blocks = (csr.nrow + BLOCK_SIZE - 1) / BLOCK_SIZE;
#pragma omp parallel for schedule(dynamic)
  for (int b = 0; b < blocks; b++) {
    // init blocks
    BlockData blocks[BLOCK_SIZE];
    unsigned nblocks = BLOCK_SIZE;
    for (int i = 0; i < BLOCK_SIZE; i++) {
      unsigned row = b * BLOCK_SIZE + i;
      blocks[i].cur = 0.0;
      blocks[i].row = row;
      blocks[i].col_index_cur = &csr.col_idx[csr.row_idx[row]];
      blocks[i].col_index_end = &csr.col_idx[csr.row_idx[row + 1]];
      blocks[i].col_index_aligned_end =
          (unsigned *)((uintptr_t)blocks[i].col_index_end & ~(uintptr_t)7);
      blocks[i].val_cur = &csr.val[csr.row_idx[blocks[i].row]];
      if (row >= csr.nrow) {
        nblocks = i;
        break;
      }
    }
    // process small
    for (unsigned bi = 0; bi < nblocks; bi++) {
      if (blocks[bi].col_index_end - blocks[bi].col_index_cur <= 8) {
        while (blocks[bi].col_index_cur < blocks[bi].col_index_end) {
          process_step(&blocks[bi], input_vec);
        }
        output_vec[blocks[bi].row] = blocks[bi].cur;

        blocks[bi] = blocks[--nblocks];
        bi--;
      }
    }
    /*// align blocks
    for (unsigned bi = 0; bi < nblocks; bi++) {
      unsigned *aligned_start =
          (unsigned *)(((uintptr_t)blocks[bi].col_index_cur + 7) &
                       ~((uintptr_t)7));
      while (blocks[bi].col_index_cur < aligned_start) {
        process_step(&blocks[bi], input_vec);
      }
    }*/

    // process remaining
    while (nblocks > 0) {
      for (unsigned bi = 0; bi < nblocks; bi++) {
        for (int kd = 0; kd < BLOCK_ADV; kd++) {

          if (blocks[bi].col_index_cur < blocks[bi].col_index_end) {
            process_step(&blocks[bi], input_vec);
          } else {
            output_vec[blocks[bi].row] = blocks[bi].cur;
            blocks[bi] = blocks[--nblocks];
            bi--;
            break;
          }
        }
      }
    }
  }
  /*for (unsigned i = 0; i < csr.nrow; ++i) {
    output_vec[i] = 0.0;
    unsigned start = csr.row_idx[i];
    unsigned aligned_start = (start + 7) & ~7;
    unsigned end = csr.row_idx[i + 1];
    unsigned aligned_end = end & ~7;
    if (aligned_start > aligned_end) {
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
  }*/
  return 0;
}

int main(int argc, char *argv[]) {
  printf("simd ilp openmp block\n");
  CSR *csr = common_read_from_file(argc, argv);

  float *input = common_generate_random_input(csr);
  float *output = (float *)malloc(sizeof(float) * csr->nrow * 2);
  int sorted = 1;
  for (unsigned i = 0; i < csr->nrow && sorted; i++) {
    unsigned cur_col = csr->col_idx[csr->row_idx[i]];
    for (unsigned j = csr->row_idx[i]; j < csr->row_idx[i + 1] && sorted; j++) {
      if (csr->col_idx[j] < cur_col) {
        sorted = 0;
        break;
      } else {
        cur_col = csr->col_idx[j];
      }
    }
  }
  if (sorted) {
    printf("is sorted\n");
  } else {
    printf("is not sorted\n");
  }

  TEST_FUNCTION(spmv_csr_simd_ilp_openmp(*csr, csr->ncol, input, output);)

  spmv_csr(*csr, csr->ncol, input, output + csr->nrow);

  if (relative_error_compare(output, output + csr->nrow, csr->nrow)) {
    printf("Error in the output\n");
    return -1;
  }

  csr_free(csr);
  free(input);
  free(output);
  printf("test passed\n\n");
  return 0;
}