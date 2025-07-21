#include <cusparse.h>
#include <cusparse_v2.h>
#include <sys/select.h>
#include <sys/time.h>
extern "C" {
#include "lib.h"
}



int main(int argc, char *argv[]) {
  printf("cusparse baseline alg 2\n");
  CSR *csr = read_from_file(argc, argv);

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
  free(csr);
  return 0;
}
