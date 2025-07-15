#include <cassert>
#include <cstdlib>
extern "C" {
#include "lib.h"
}
#include <math.h>
#include <stdio.h>
#include <sys/select.h>
#include <sys/time.h>
#define ROWS (1 << 13)
#define COLS (1 << 13)
#define NNZ (1 << 24)

#define WARMUPS 2
#define REPS 5

#define BLOCK_SIZE 256
#define SHARED_DATA_BLOCK 600000

#define CHECK_CUDA(call)                                                       \
  if ((call) != cudaSuccess) {                                                 \
    fprintf(stderr, "CUDA error at %s:%u %u\n", __FILE__, __LINE__, call);     \
    exit(1);                                                                   \
  }

__global__ void spmv_csr_gpu_kernel_chunks(const float *__restrict__ vals,
                                           const unsigned *__restrict__ row_idx,
                                           const unsigned *__restrict__ col_idx,
                                           const float *__restrict__ input_vec,
                                           float *output_vec, unsigned nrow,
                                           unsigned ncol, unsigned nnz) {
  // __shared__ float shared_vals[SHARED_DATA_BLOCK];
  ;
  unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
  // 1) load what we could use (maybe not because of broadcasting)
  // 2) advance and repeat
  // 3) write to output
  if (row >= nrow) {
    return;
  }

  float cached_output = 0.0;

  unsigned cur_pos = row_idx[row];
  unsigned cur_row_end = row_idx[row + 1];

  __syncthreads();
  unsigned nblocks = (nnz + SHARED_DATA_BLOCK - 1) / SHARED_DATA_BLOCK;
  for (unsigned cur_block = 0; cur_block < nblocks; cur_block++) {

    // load data into shared memory (from cur_block*SHARED_DATA_BLOCK to
    // (cur_block+1)*SHARED_DATA_BLOCK)
    unsigned block_start = cur_block * SHARED_DATA_BLOCK;
    unsigned block_end = block_start + SHARED_DATA_BLOCK;
    /*for(unsigned j = threadIdx.x; j < SHARED_DATA_BLOCK; j += blockDim.x) {
      unsigned idx = block_start + j;
      if (idx < nnz) {
        shared_vals[j] = vals[idx];
      } else {
        shared_vals[j] = 0.0f; // Fill with zero if out of bounds
      }
    }*/
    __syncthreads();
    while (cur_pos < cur_row_end && col_idx[cur_pos] < block_end) {
      unsigned col = col_idx[cur_pos];
      cached_output += vals[cur_pos] * input_vec[col];
      cur_pos++;
    }
    __syncthreads();
  }

  // Write the result to the output vector
  output_vec[row] = cached_output;
}
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

void dummy_launcher(CSR *gpu_csr, float *input_vec_gpu, float *output_gpu) {
  unsigned nblocks = (gpu_csr->nrow + BLOCK_SIZE - 1) / BLOCK_SIZE;
  spmv_csr_gpu_kernel_chunks<<<nblocks, BLOCK_SIZE>>>(
      gpu_csr->val, gpu_csr->row_idx, gpu_csr->col_idx, input_vec_gpu,
      output_gpu, gpu_csr->nrow, gpu_csr->ncol, gpu_csr->nnz);
  CHECK_CUDA(cudaDeviceSynchronize());
}
int spmv_csr_gpu_chunks(CSR *csr, unsigned n, float *input_vec,
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

  TEST_FUNCTION(dummy_launcher(gpu_csr, input_vec_gpu, output_gpu));

  CHECK_CUDA(cudaMemcpy(output_vec, output_gpu, sizeof(float) * gpu_csr->nrow,
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(input_vec_gpu));
  CHECK_CUDA(cudaFree(output_gpu));
  free_csr_gpu(gpu_csr);
  return 0;
}

int write_bin_to_file(CSR *csr, const char *filename) {
  FILE *output = fopen(filename, "wb");
  if (output == NULL) {
    fprintf(stderr, "Error opening file for writing: %s\n", filename);
    return -1;
  }
  fwrite(&csr->nrow, sizeof(unsigned), 1, output);
  fwrite(&csr->ncol, sizeof(unsigned), 1, output);
  fwrite(&csr->nnz, sizeof(unsigned), 1, output);
  fwrite(csr->row_idx, sizeof(unsigned), csr->nrow + 1, output);
  fwrite(csr->col_idx, sizeof(unsigned), csr->nnz, output);
  fwrite(csr->val, sizeof(float), csr->nnz, output);
  fclose(output);
  return 0;
}
int read_bin_to_csr(const char *filename, CSR *csr) {
  FILE *input = fopen(filename, "rb");
  if (input == NULL) {
    fprintf(stderr, "Error opening file for reading: %s\n", filename);
    return -1;
  }
  fread(&csr->nrow, sizeof(unsigned), 1, input);
  fread(&csr->ncol, sizeof(unsigned), 1, input);
  fread(&csr->nnz, sizeof(unsigned), 1, input);
  csr->row_idx = (unsigned *)malloc((csr->nrow + 1) * sizeof(unsigned));
  csr->col_idx = (unsigned *)malloc(csr->nnz * sizeof(unsigned));
  csr->val = (float *)malloc(csr->nnz * sizeof(float));
  fread(csr->row_idx, sizeof(unsigned), csr->nrow + 1,  input);
  fread(csr->col_idx, sizeof(unsigned), csr->nnz, input);
  fread(csr->val, sizeof(float), csr->nnz, input);
  fclose(input);
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
    //coo_generate_random(coo, ROWS, COLS, NNZ);
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

  spmv_csr_gpu_chunks(csr, csr->ncol, input, output); //, tmp
  spmv_csr(*csr, csr->ncol, input, output + csr->nrow);

  if (relative_error_compare(output, output + csr->nrow, csr->nrow)) {
    printf("Error in the output\n");
    return -1;
  }

  coo_free(coo);
  csr_free(csr);
  free(input);
  free(output);
  printf("test passed\n\n");
  return 0;
}
