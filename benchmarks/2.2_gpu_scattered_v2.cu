#include <cassert>
extern "C" {
#include "lib.h"
}
#include <math.h>
#include <stdio.h>
#include <sys/select.h>
#include <sys/time.h>
#include <unistd.h>

// how many threads per block
#define BLOCK_THREADS (96)

// size of data_block, so how many consecutive elements to process in a single
// block
#define DATA_BLOCK (384)
#define WRITE_OUT_BLOCKS 8

__device__ inline unsigned normal_upper_bound(const unsigned *__restrict__ arr,
                                              int size, unsigned key) {
  unsigned left = 0;
  unsigned right = size;
  while (left + 1 < right) {
    // printf("left %u right %u\n", left, right);
    unsigned mid = (right + left) >> 1;
    if (__ldg(arr + mid) <= key)
      left = mid;
    else
      right = mid;
  }
  return left;
}

__global__ void spmv_csr_gpu_kernel_nnz(const float *__restrict__ val,
                                        const unsigned *__restrict__ row_idx,
                                        const unsigned *__restrict__ col_idx,
                                        const float *__restrict__ input_vec,
                                        float *output_vec, unsigned nrow,
                                        unsigned ncol, unsigned nnz) {
  __shared__ unsigned shared_rows_idx[DATA_BLOCK];
  __shared__ float contributions[DATA_BLOCK];
  // Use interleaved assignment instead of consecutive
  /*unsigned stride = gridDim.x;
  unsigned block_id = blockIdx.x;

  // Calculate interleaved block range
  unsigned elements_per_block = (nnz + stride - 1) / stride;
  unsigned block_start = block_id * elements_per_block;
  unsigned block_end = min(block_start + elements_per_block, nnz);

  unsigned assigned_start = block_start+DATA_BLOCK*threadIdx.x/BLOCK_THREADS;
  unsigned assigned_end =
  min(block_start+DATA_BLOCK*(threadIdx.x+1)/BLOCK_THREADS, block_end);*/

  // for (unsigned i = 0; i < csr.nrow; ++i) {
  unsigned block_start = blockIdx.x * DATA_BLOCK;
  unsigned block_end = min(block_start + DATA_BLOCK, nnz);
  /// build the shared memory with the row_idx
  unsigned assigned_start =
      block_start + DATA_BLOCK * threadIdx.x / BLOCK_THREADS;
  unsigned assigned_end = min(
      block_start + DATA_BLOCK * (threadIdx.x + 1) / BLOCK_THREADS, block_end);

  // Async copy from global to shared memory
  /*auto block = cg::this_thread_block();
  cg::memcpy_async(block, shared_rows_idx, row_idx + assigned_start,
  sizeof(unsigned) * (assigned_end - assigned_start)); cg::memcpy_async(block,
  contributions, val + assigned_start, sizeof(float) * (assigned_end -
  assigned_start)); cg::wait(block);*/
  unsigned row = 0;
  for (unsigned i = assigned_start; i < assigned_end;) {
    row = normal_upper_bound(row_idx, nrow, i);
    /*unsigned other_row=normal_upper_bound(row_idx, nrow, i);
    if(row!=other_row){
      printf("%u %u\n", row, other_row);
      assert(row==other_row);
    }*/
    unsigned row_end = min(row_idx[row + 1], assigned_end);
    for (unsigned j = i; j < row_end; j++) {
      shared_rows_idx[j - block_start] = row;
    }
    i = row_end;
  }

  __syncthreads();

  unsigned start = block_start + threadIdx.x;
  if (start < block_end) {
    for (unsigned i = start; i < block_end; i += BLOCK_THREADS) {
      contributions[i - block_start] = val[i] * input_vec[col_idx[i]];
    }
  }

  __syncthreads();

  // accumulate all contributions and write them in a single atomic operation
  if (threadIdx.x < WRITE_OUT_BLOCKS) {
    unsigned assigned_start =
        block_start + (DATA_BLOCK * threadIdx.x / WRITE_OUT_BLOCKS);
    unsigned assigned_end =
        min(block_start + (DATA_BLOCK * (threadIdx.x + 1) / WRITE_OUT_BLOCKS),
            block_end);
    float contrib = 0.0;
    unsigned prev_row = shared_rows_idx[0];
    bool first = true;

    for (unsigned i = assigned_start; i < assigned_end; i++) {
      if (shared_rows_idx[i - block_start] != prev_row) {
        if (first) {
          atomicAdd(&output_vec[prev_row], contrib);
        } else {
          first = false;
          output_vec[prev_row] = contrib;
        }
        // atomicAdd(&output_vec[prev_row], contrib);
        contrib = 0.0;
        prev_row = shared_rows_idx[i - block_start];
      }
      // contrib += contributions[i];
      contrib += contributions[i - block_start];
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
  // atomicAdd(&output_vec[prev_row], cur);
}

void dummy_launcher(CSR *csr, float *input_vec, float *output_vec) {
  cudaMemset(output_vec, 0, sizeof(float) * csr->nrow);
  unsigned int nblocks = (csr->nnz + DATA_BLOCK - 1) / DATA_BLOCK;
  spmv_csr_gpu_kernel_nnz<<<nblocks, BLOCK_THREADS>>>(
      csr->val, csr->row_idx, csr->col_idx, input_vec, output_vec, csr->nrow,
      csr->ncol, csr->nnz);
  /*cudaEvent_t kernel_done;
  cudaEventCreate(&kernel_done);
  cudaEventRecord(kernel_done);
  while (cudaEventQuery(kernel_done) == cudaErrorNotReady) {
    usleep(100); // Sleep for 100 microseconds
  }
  cudaEventDestroy(kernel_done);*/
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
  CSR *csr = common_read_from_file(argc, argv);

  float *input = common_generate_random_input(csr);

  float *output = (float *)malloc(sizeof(float) * csr->nrow * 2);

  spmv_csr_gpu_nnz(csr, csr->ncol, input, output); //, tmp
  spmv_csr(*csr, csr->ncol, input, output + csr->nrow);

  if (relative_error_compare(output, output + csr->nrow, csr->nrow)) {
    printf("Error in the output\n");
    return 0;
  }

  csr_free(csr);
  free(input);
  free(output);
  printf("test passed\n\n");
  return 0;
}
