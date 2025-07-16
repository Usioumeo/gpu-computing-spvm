#include <cusparse.h>
#include <cusparse_v2.h>
#include <sys/select.h>
#include <sys/time.h>
#define USE_CUDA
extern "C" {
#include "lib.h"
}

#define ROWS (1 << 13)
#define COLS (1 << 13)
#define NNZ (1 << 24)

#define WARMUPS 40
#define REPS 500

int main(int argc, char *argv[]) {
  printf("cusparse baseline alg 2\n");
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

  float *rand_vec;// = (float *)malloc(sizeof(float) * csr->ncol);
  cudaMallocManaged(&rand_vec, sizeof(float) * csr->ncol);
  float *output; //= (float *)malloc(sizeof(float) * csr->ncol * 2);
  cudaMallocManaged(&output, sizeof(float) * csr->nrow * 2);
  for (unsigned i = 0; i < csr->ncol; i++) {
    rand_vec[i] = (float)(rand() % 2001 - 1000) * 0.001;
  }
  // cuSPARSE handle and descriptors
  cusparseHandle_t handle;
  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t input_vec, output_vec;
  void *dBuffer = nullptr;
  size_t bufferSize = 0;

  cusparseCreate(&handle);

  // Create sparse matrix A in CSR format
  cusparseCreateCsr(&matA, csr->nrow, csr->ncol, csr->nnz, csr->row_idx,
                    csr->col_idx, csr->val, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

  // Create dense vectors
  cusparseCreateDnVec(&input_vec, csr->nrow, rand_vec, CUDA_R_32F);
  cusparseCreateDnVec(&output_vec, csr->ncol, output, CUDA_R_32F);

  // Prepare parameters for multiplication
  float alpha = 1.0f, beta = 0.0f;

  // Query buffer size for SpMV
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                          matA, input_vec, &beta, output_vec, CUDA_R_32F,
                          CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
  cudaMalloc(&dBuffer, bufferSize);

  // Timed repetitions

  TEST_FUNCTION(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                             matA, input_vec, &beta, output_vec, CUDA_R_32F,
                             CUSPARSE_SPMV_CSR_ALG2, dBuffer);
                cudaDeviceSynchronize();)
  spmv_csr(*csr,  csr->ncol, rand_vec, output +  csr->nrow);

  if (relative_error_compare(output, output + csr->nrow, csr->nrow)) {
    printf("Error in the output\n");
    return -1;
  }
  // Cleanup
  cusparseDestroyDnVec(input_vec);
  cusparseDestroyDnVec(output_vec);
  cusparseDestroySpMat(matA);
  cusparseDestroy(handle);
  cudaFree(dBuffer);
  cudaFree(rand_vec);
  cudaFree(output);

  return 0;
}
