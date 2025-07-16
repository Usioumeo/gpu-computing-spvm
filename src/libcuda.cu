extern "C" {
#define USE_CUDA
#include "lib.h"
CSR *copy_csr_to_gpu(CSR *csr) {
  // Move CSR data to GPU
  float *d_val;
  unsigned *d_col_idx, *d_row_idx;
  CSR *ret = (CSR *)malloc(sizeof(CSR));

  CHECK_CUDA(cudaMalloc(&d_val, csr->nnz * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_col_idx, csr->nnz * sizeof(unsigned)));
  CHECK_CUDA(cudaMalloc(&d_row_idx, (csr->nrow + 1) * sizeof(unsigned)));

  CHECK_CUDA(cudaMemcpy(d_val, csr->val, csr->nnz * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_col_idx, csr->col_idx, csr->nnz * sizeof(unsigned),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_row_idx, csr->row_idx,
                        (csr->nrow + 1) * sizeof(unsigned),
                        cudaMemcpyHostToDevice));

  ret->val = d_val;
  ret->col_idx = d_col_idx;
  ret->row_idx = d_row_idx;
  ret->ncol = csr->ncol; // Keep the number of columns
  ret->nrow = csr->nrow; // Keep the number of rows
  ret->nnz = csr->nnz;   // Keep the number of non-zero

  return ret;
}

void free_csr_gpu(CSR *csr) {
  // Free GPU memory
  CHECK_CUDA(cudaFree(csr->val));
  CHECK_CUDA(cudaFree(csr->col_idx));
  CHECK_CUDA(cudaFree(csr->row_idx));
  free(csr);
}
}