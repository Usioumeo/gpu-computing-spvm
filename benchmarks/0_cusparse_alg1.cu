#include <cusparse.h>
#include <cusparse_v2.h>
#include <sys/select.h>
#include <sys/time.h>
extern "C" {
#include "lib.h"
}

#define ROWS (1 << 13)
#define COLS (1 << 13)
#define NNZ (1 << 24)

#define WARMUPS 0
#define REPS 5

int spmv_csr_gpu_cusparse(CSR *csr, unsigned n, float *input_vec,
                          float *output_vec) {
  if (n != csr->ncol) {
    return 1;
  }
  CSR *gpu_csr = copy_csr_to_gpu(csr);

  float *input_vec_gpu, *output_gpu;
  CHECK_CUDA(cudaMalloc(&input_vec_gpu, sizeof(float) * csr->ncol));
  CHECK_CUDA(cudaMemcpy(input_vec_gpu, input_vec, sizeof(float) * csr->ncol,
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&output_gpu, sizeof(float) * gpu_csr->nrow));

  // now cusparse handling
  //  cuSPARSE handle and descriptors
  cusparseHandle_t handle;
  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t input_vec_cuda, output_vec_cuda;
  void *dBuffer = nullptr;
  size_t bufferSize = 0;

  cusparseCreate(&handle);

  // Create sparse matrix A in CSR format
  cusparseCreateCsr(&matA, csr->nrow, gpu_csr->ncol, gpu_csr->nnz, gpu_csr->row_idx,
                    gpu_csr->col_idx, gpu_csr->val, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

  // Create dense vectors
  cusparseCreateDnVec(&input_vec_cuda, csr->ncol, input_vec_gpu, CUDA_R_32F);
  cusparseCreateDnVec(&output_vec_cuda, csr->nrow, output_gpu, CUDA_R_32F);

  // Prepare parameters for multiplication
  float alpha = 1.0f, beta = 0.0f;

  // Query buffer size for SpMV
  cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                          matA, input_vec_cuda, &beta, output_vec_cuda,
                          CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
  cudaMalloc(&dBuffer, bufferSize);
  TEST_FUNCTION(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                             matA, input_vec_cuda, &beta, output_vec_cuda,
                             CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, dBuffer);
                cudaDeviceSynchronize();)

  // end cusparse handling
  CHECK_CUDA(cudaMemcpy(output_vec, output_gpu, sizeof(float) * gpu_csr->nrow,
                        cudaMemcpyDeviceToHost));
  // Cleanup
  cusparseDestroyDnVec(input_vec_cuda);
  cusparseDestroyDnVec(output_vec_cuda);
  cusparseDestroySpMat(matA);
  cusparseDestroy(handle);
  CHECK_CUDA(cudaFree(input_vec_gpu));
  CHECK_CUDA(cudaFree(output_gpu));
  cudaFree(dBuffer);
  //cudaFree(input_);
  free_csr_gpu(gpu_csr);
  return 0;
}

int main(int argc, char *argv[]) {
  COO *coo = coo_new();
  CSR *csr = csr_new();
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
    coo_to_csr(coo, csr);
    write_bin_to_file(csr, "tmp.bin");
  } else {
    // coo_generate_random(coo, ROWS, COLS, NNZ);
    read_bin_to_csr("tmp.bin", csr);
  }

  printf("csr->nrow %u csr->ncol %u csr->nnz %u\n", csr->nrow, csr->ncol,
         csr->nnz);

  float *input = (float *)malloc(sizeof(float) * csr->ncol);
  // cudaMallocHost(&rand_vec_host, sizeof(float)*COLS);
  for (unsigned i = 0; i < csr->ncol; i++) {
    input[i] = (float)(rand() % 2001 - 1000) * 0.001;
  }

  float *output = (float *)malloc(sizeof(float) * csr->nrow * 2);

  // Timed repetitions
  spmv_csr_gpu_cusparse(csr, csr->ncol, input, output);
  spmv_csr(*csr, csr->ncol, input, output + csr->nrow);

  if (relative_error_compare(output, output + csr->nrow, csr->nrow)) {
    printf("Error in the output\n");
    return -1;
  }

  
  cudaFree(output);

  return 0;
}
