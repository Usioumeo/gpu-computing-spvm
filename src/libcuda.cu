
extern "C" {
  #include "lib.h"
void csr_reserve(CSR *csr, unsigned nnz, unsigned nrow) {
  if (csr->nnz < nnz || csr->nrow < nrow || csr->val == NULL ||
      csr->col_idx == NULL || csr->row_idx == NULL) {
    // resize csr arrays
    if (csr->row_idx != NULL) {
      cudaFree(csr->row_idx);
    }
    cudaMallocManaged(&csr->row_idx, (nrow + 1) * sizeof(unsigned));
    if (csr->col_idx != NULL) {
      cudaFree(csr->col_idx);
    }
    cudaMallocManaged(&csr->col_idx, nnz * sizeof(unsigned));
    if (csr->val != NULL) {
      cudaFree(csr->val);
    }
    cudaMallocManaged(&csr->val, nnz * sizeof(float));
    //(unsigned *)realloc(csr->row_idx, (nrow + 1) * sizeof(unsigned));
    // csr->col_idx = (unsigned *)realloc(csr->col_idx, nnz * sizeof(unsigned));
    // csr->val = (float *)realloc(csr->val, nnz * sizeof(float));
  }
  csr->nnz = nnz;
  csr->nrow = nrow;
}

// Function to free the memory allocated for CSR matrix
// IT ALSO FREES THE POINTER
void csr_free(CSR *csr) {

  cudaFree(csr->row_idx);
  cudaFree(csr->col_idx);
  cudaFree(csr->val);
  csr->row_idx = NULL;
  csr->col_idx = NULL;
  csr->val = NULL;
  cudaFree(csr);
}

// Function to create a new empty CSR matrix
CSR *csr_new() {
  CSR *csr; //= (CSR *)malloc(sizeof(CSR));
  cudaMallocManaged(&csr, sizeof(CSR));
  csr->row_idx = NULL;
  csr->col_idx = NULL;
  csr->val = NULL;
  csr->nnz = 0;
  csr->nrow = 0;
  csr->ncol = 0;
  return csr;
}
}