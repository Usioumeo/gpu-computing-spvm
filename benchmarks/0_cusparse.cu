#include <cusparse.h>
#include <cusparse_v2.h>
#include <sys/select.h>
#include <sys/time.h>

extern "C" {
#include "lib.h"
}

#define ROWS (1<<13)
#define COLS (1<<13)
#define NNZ (1<<24)

#define WARMUPS 40
#define REPS 500

int main() {
    COO *coo = coo_new();
  // cudaMallocManaged(&coo, sizeof(COO));
  coo_generate_random(coo, ROWS, COLS, NNZ);
  CSR *csr = csr_new();
  coo_to_csr(coo, csr);

  // Assign CSR arrays to device pointers for cuSPARSE

  float *rand_vec; // = (float * )malloc(sizeof(float)*COLS);
  cudaMallocManaged(&rand_vec, COLS * sizeof(float));
  float *output; //= (float*)malloc(sizeof(float)*COLS*(REPS+1));
  cudaMallocManaged(&output, COLS * 2 * sizeof(float));
  memset(output, 0, sizeof(float) * COLS * 2);
  for (unsigned i = 0; i < COLS; i++) {
    rand_vec[i] = (float)(rand() % 2001 - 1000) * 0.001;
  }

    /*CHECK_CUDA(cudaMalloc((void**)&dA_values,     nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dA_columns,    nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dA_row_offsets,(rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dx,            cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dy,            rows * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA_values, hA_values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_row_offsets, hA_row_offsets, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dx, hx, cols * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dy, hy, rows * sizeof(float), cudaMemcpyHostToDevice));*/

    // cuSPARSE handle and descriptors
    cusparseHandle_t handle;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t input_vec, output_vec;
    void* dBuffer = nullptr;
    size_t bufferSize = 0;

    cusparseCreate(&handle);

    // Create sparse matrix A in CSR format
    cusparseCreateCsr(&matA, csr->nrow, csr->ncol, csr->nnz,
                                     csr->row_idx, csr->col_idx, csr->val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // Create dense vectors
    cusparseCreateDnVec(&input_vec, csr->nrow, rand_vec, CUDA_R_32F);
    cusparseCreateDnVec(&output_vec, csr->ncol, output, CUDA_R_32F);

    // Prepare parameters for multiplication
    float alpha = 1.0f, beta = 0.0f;

    // Query buffer size for SpMV
    cusparseSpMV_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        input_vec,
        &beta,
        output_vec,
        CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &bufferSize
    );
    cudaMalloc(&dBuffer, bufferSize);


    // Timed repetitions
    
        TEST_FUNCTION(cusparseSpMV(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            matA,
            input_vec,
            &beta,
            output_vec,
            CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            dBuffer
        );
        cudaDeviceSynchronize();)
    spmv_csr(*csr, COLS, rand_vec, &output[COLS]);
    // Check results
    if(relative_error_compare(output, output+csr->ncol, csr->ncol)) {
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
