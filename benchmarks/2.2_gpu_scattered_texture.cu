#include <cassert>
extern "C" {
#include "lib.h"
}
#include <math.h>
#include <stdio.h>
#include <sys/select.h>
#include <sys/time.h>
#include <nvtx3/nvToolsExt.h>

#define WRITE_OUT_BLOCKS 8
//how many threads per block
#define BLOCK_THREADS (96)

// size of data_block, so how many consegutive elements to process in a single block
#define DATA_BLOCK (384)

__device__ inline unsigned upper_bound(cudaTextureObject_t arr, int size, unsigned key) {
  int left = 0;
  int right = size;
  #pragma unroll 4
  while (left + 1 < right) {
    // printf("left %u right %u\n", left, right);
    int mid = (right + left)>>1;
    if (tex1Dfetch<unsigned>(arr, mid) <= key)
      left = mid;
    else
      right = mid;
  }
  return left;
}

__device__ inline unsigned upper_bound_next(cudaTextureObject_t arr, int size, unsigned key, unsigned prev) {
  int left = prev;
  int right=prev+1;
  while(right<size&& tex1Dfetch<unsigned>(arr, right)<=key) {
    right=(right|(right-1))+1;
  }
  right = min(right, size);
  while (left + 1 < right) {
    // printf("left %u right %u\n", left, right);
    int mid = (right + left)>>1;
    if (tex1Dfetch<unsigned>(arr, mid) <= key)
      left = mid;
    else
      right = mid;
  }
  return left;
}
__global__ void spmv_csr_gpu_kernel_nnz( const float* __restrict__ val, cudaTextureObject_t row_idx, const unsigned* __restrict__ col_idx,
                                           cudaTextureObject_t input_tex, float *output_vec, unsigned nrow, unsigned ncol, unsigned nnz){
  __shared__ unsigned shared_rows_idx[DATA_BLOCK];
  __shared__ float contributions[DATA_BLOCK];
                                            // for (unsigned i = 0; i < csr.nrow; ++i) {
  unsigned block_start = blockIdx.x * DATA_BLOCK;
  unsigned block_end = min(block_start + DATA_BLOCK, nnz);
  ///build the shared memory with the row_idx
  unsigned assigned_start = block_start+(DATA_BLOCK*threadIdx.x)/BLOCK_THREADS;
  unsigned assigned_end = min(block_start+(DATA_BLOCK*(threadIdx.x+1))/BLOCK_THREADS, block_end);
  //get rows
  unsigned row=0;
  for(unsigned i=assigned_start; i<assigned_end; ) {
    row = upper_bound(row_idx, nrow, i);
    //unsigned row_end = i+1;
    unsigned row_end=min(tex1Dfetch<unsigned>(row_idx, row+1), assigned_end);
    for(unsigned j=i; j < row_end; j++) {
      shared_rows_idx[j-block_start] = row;
    }
    i=row_end;
  }
  __syncthreads();

  //compute
  unsigned start = block_start+ threadIdx.x;
  if (start < block_end) {//shared_rows_idx[start - block_start];
    #pragma unroll 10
    for(unsigned i=start; i<block_end; i+= BLOCK_THREADS) {
        contributions[i-block_start]= val[i] * tex1Dfetch<float>(input_tex, col_idx[i]);
    }

  }
  
  __syncthreads();
  
  //accumulate all contributions and write them in the least amount of atomic operations
  if(threadIdx.x<WRITE_OUT_BLOCKS){
    unsigned assigned_start = block_start+(DATA_BLOCK*threadIdx.x)/WRITE_OUT_BLOCKS;
    unsigned assigned_end = min(block_start+(DATA_BLOCK*(threadIdx.x+1))/WRITE_OUT_BLOCKS, block_end);
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
        contrib = 0.0;
        prev_row = shared_rows_idx[i-block_start];
      }
      contrib+= contributions[i-block_start];
    }
    atomicAdd(&output_vec[prev_row], contrib);
  }
}

void dummy_launcher(CSR *csr, cudaTextureObject_t input_tex, float *output_vec, cudaTextureObject_t row_idx) {


  CHECK_CUDA(cudaMemset(output_vec, 0, sizeof(float) * csr->nrow));
  unsigned int nblocks = (csr->nnz + DATA_BLOCK - 1) / DATA_BLOCK;
  spmv_csr_gpu_kernel_nnz<<<nblocks, BLOCK_THREADS>>>(
      csr->val, row_idx, csr->col_idx, input_tex, output_vec, csr->nrow,
      csr->ncol, csr->nnz);
  //cpu_busy_loop();
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
  printf("nrow %u ncol %u nnz %u\n", csr->nrow, csr->ncol, csr->nnz);
  CHECK_CUDA(cudaMalloc(&output_gpu, sizeof(float) * csr->nrow));
  //create texture object for input
     // Create texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = input_vec_gpu;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // 32-bit float
    resDesc.res.linear.sizeInBytes = csr->ncol * sizeof(float);
    
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    
    cudaTextureObject_t input_tex = 0;
    CHECK_CUDA(cudaCreateTextureObject(&input_tex, &resDesc, &texDesc, NULL));
    

    /// CREATE TEXTURE OBJECT FOR ROW IDX
cudaResourceDesc resDesc2;
    memset(&resDesc2, 0, sizeof(resDesc2));
    resDesc2.resType = cudaResourceTypeLinear;
    resDesc2.res.linear.devPtr = gpu_csr->row_idx;
    resDesc2.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    resDesc2.res.linear.desc.x = 32; // 32-bit float
    resDesc2.res.linear.sizeInBytes = (csr->nrow+1) * sizeof(unsigned);

    cudaTextureDesc texDesc2;
    memset(&texDesc2, 0, sizeof(texDesc2));
    texDesc2.addressMode[0] = cudaAddressModeClamp;
    texDesc2.filterMode = cudaFilterModePoint;
    texDesc2.readMode = cudaReadModeElementType;
    texDesc2.normalizedCoords = 0;

    cudaTextureObject_t row_idx_tex = 0;
    CHECK_CUDA(cudaCreateTextureObject(&row_idx_tex, &resDesc2, &texDesc2, NULL));







  TEST_FUNCTION(dummy_launcher(gpu_csr, input_tex, output_gpu, row_idx_tex));
  CHECK_CUDA(cudaDestroyTextureObject(input_tex));
  CHECK_CUDA(cudaDestroyTextureObject(row_idx_tex));
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
