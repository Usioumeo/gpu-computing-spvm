#include <cassert>
#define USE_CUDA
extern "C" {
#include "lib.h"
}
#include <math.h>
#include <stdio.h>
#include <sys/select.h>
#include <sys/time.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#define WARMUPS 0
#define REPS 2

//how many threads per block
#define BLOCK_THREADS 128

// size of data_block, so how many consegutive elements to process in a single block
#define DATA_BLOCK 384

__device__ inline unsigned upper_bound(const unsigned *__restrict__ arr, int size, unsigned key) {;
  int left = 0;
  int right = size;
  while (left + 1 < right) {
    // printf("left %u right %u\n", left, right);
    int mid = (right + left)>>1;
    if (__ldg(arr + mid) <= key)
      left = mid;
    else
      right = mid;
  }
  return left;
}


__global__ void spmv_csr_gpu_kernel_nnz( const float* __restrict__ val, const unsigned * __restrict__ row_idx, const unsigned* __restrict__ col_idx,
                                           const float* __restrict__ input_vec, float *output_vec, unsigned nrow, unsigned ncol, unsigned nnz){
  __shared__ unsigned shared_rows_idx[DATA_BLOCK];
  __shared__ float contributions[DATA_BLOCK];

                                            // for (unsigned i = 0; i < csr.nrow; ++i) {
  unsigned block_start = blockIdx.x * DATA_BLOCK;
  unsigned block_end = min(block_start + DATA_BLOCK, nnz);
  ///build the shared memory with the row_idx
  unsigned assigned_start = block_start+(DATA_BLOCK/BLOCK_THREADS)*threadIdx.x;
  unsigned assigned_end = min(block_start+(DATA_BLOCK/BLOCK_THREADS)*(threadIdx.x+1), block_end);
// Async copy from global to shared memory
    /*auto block = cg::this_thread_block();
    cg::memcpy_async(block, shared_rows_idx, row_idx + assigned_start, sizeof(unsigned) * (assigned_end - assigned_start));
    cg::memcpy_async(block, contributions, val + assigned_start, sizeof(float) * (assigned_end - assigned_start));
    cg::wait(block);*/
  for(unsigned i=assigned_start; i<assigned_end; ) {
    unsigned row = upper_bound(row_idx, nrow, i);
    unsigned row_end=min(row_idx[row+1], assigned_end);
    for(unsigned j=i; j < row_end; j++) {
      shared_rows_idx[j-block_start] = row;
    }
    i=row_end;
    /*while(i< assigned_end && i < row_idx[row+1]) {
      shared_rows_idx[i-block_start] = row;
      i++;
    }*/
  }
  /*for(unsigned i=block_start+threadIdx.x; i<block_end; i+=BLOCK_THREADS) {
    unsigned row = upper_bound(row_idx, nrow, i);
    shared_rows_idx[i-block_start] = row;
  }*/

  __syncthreads();

  unsigned start = block_start+ threadIdx.x;
  if (start < block_end) {
    //unsigned prev_row = upper_bound(row_idx, nrow, start);
    unsigned prev_row = shared_rows_idx[start - block_start];
    
    for(unsigned i=start; i<block_end; i+= BLOCK_THREADS) {
        contributions[i-block_start]= val[i] * input_vec[col_idx[i]];
        //unsigned cur_row = shared_rows_idx[i-block_start];
        //atomicAdd(&output_vec[cur_row], contributions[i-block_start]);
        //unsigned local_idx = i - block_start;
        //unsigned row = shared_rows_idx[local_idx];
        //atomicAdd(&output_vec[row], contributions[i-block_start]);
        /*if (row != prev_row) {
          
          contributions[local_idx-BLOCK_THREADS] = contribution;
          prev_row = row;
          cur = 0.0;
        }*/
    }
    //contributions[i-block_start-BLOCK_THREADS] = contribution;

  }
  
  __syncthreads();
  #define WRITE_OUT_BLOCKS 8
  //accumulate all contributions and write them in a single atomic operation
  if(threadIdx.x<WRITE_OUT_BLOCKS){
    unsigned assigned_start = block_start+(DATA_BLOCK/WRITE_OUT_BLOCKS)*threadIdx.x;
    unsigned assigned_end = min(block_start+(DATA_BLOCK/WRITE_OUT_BLOCKS)*(threadIdx.x+1), block_end);
    float contrib = 0.0;
    unsigned prev_row = shared_rows_idx[0];
    bool first = true;
    for(unsigned i=assigned_start; i<assigned_end; i++) {
      if (shared_rows_idx[i-block_start] != prev_row) {
        if (first) {
          atomicAdd(&output_vec[prev_row], contrib);
        } else {
          first = false;
          output_vec[prev_row] = contrib;
        }
        //atomicAdd(&output_vec[prev_row], contrib);
        contrib = 0.0;
        prev_row = shared_rows_idx[i-block_start];
      }
      //contrib += contributions[i];
      contrib+= contributions[i-block_start];
    }
    atomicAdd(&output_vec[prev_row], contrib);
  }

  /*for(int i=start; i<block_end; i+= BLOCK_THREADS) {
      float contribution = val[i] * input_vec[col_idx[i]];
      unsigned local_idx = i - block_start;
      unsigned row = shared_rows_idx[local_idx];
      //atomicAdd(&output_vec[row], contribution);
      //output_vec[row] += contribution;
  }*/
  //atomicAdd(&output_vec[prev_row], cur);
}

void dummy_launcher(CSR *csr, float *input_vec, float *output_vec) {
  cudaMemset(output_vec, 0, sizeof(float) * csr->nrow);
  unsigned int nblocks = (csr->nnz + DATA_BLOCK - 1) / DATA_BLOCK;

  printf("nblocks %u\n", nblocks);
  spmv_csr_gpu_kernel_nnz<<<nblocks, BLOCK_THREADS>>>(
      csr->val, csr->row_idx, csr->col_idx, input_vec, output_vec, csr->nrow,
      csr->ncol, csr->nnz);
  CHECK_CUDA(cudaDeviceSynchronize());
}

int spmv_csr_gpu_nnz(CSR *csr, unsigned n, float *input_vec,
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

  spmv_csr_gpu_nnz(csr, csr->ncol, input, output); //, tmp
  spmv_csr(*csr, csr->ncol, input, output + csr->nrow);

  if (relative_error_compare(output, output + csr->nrow, csr->nrow)) {
    printf("Error in the output\n");
    return 0;
  }

  coo_free(coo);
  csr_free(csr);
  free(input);
  free(output);
  printf("test passed\n\n");
  return 0;
}
