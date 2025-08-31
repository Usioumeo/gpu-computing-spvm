
#include <cassert>
extern "C" {
#include "lib.h"
}
#include <math.h>
#include <stdio.h>
#include <sys/select.h>
#include <sys/time.h>

#define BLOCK_SIZE 32
#define DATA_BLOCK (16)

__device__ unsigned upper_bound(const unsigned *arr, int size, unsigned key) {
  int left = 0;
  int right = size;
  while (left + 1 < right) {
    // printf("left %u right %u\n", left, right);
    int mid = (right + left) / 2;
    if (arr[mid] <= key)
      left = mid;
    else
      right = mid;
  }
  return left;
}

__global__ void spmv_csr_gpu_kernel_chunks(const float *__restrict__ val,
                                           const unsigned *__restrict__ row_idx,
                                           const unsigned *__restrict__ col_idx,
                                           cudaTextureObject_t input_tex,
                                           float *output_vec, unsigned nrow,
                                           unsigned ncol, unsigned nnz) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned starting_point = i * DATA_BLOCK;
  if (starting_point >= nnz) {
    return;
  }
  unsigned row = upper_bound(row_idx, nrow, starting_point);
  unsigned next_row_pos = row_idx[row + 1];
  float cur_val = 0.0;
  unsigned end_point = starting_point + DATA_BLOCK;
  if (end_point > nnz) {
    end_point = nnz;
  }
  const unsigned *next_row_idx = &row_idx[row + 1];
  for (unsigned j = starting_point; j < end_point; j++) {
    if (j == next_row_pos) {
      // printf("row %u += %f\n", row, cur_val);
      // exit(1);
      //  THIS MUST BE DONE IN A ATOMIC WAY
      // output_vec[row]+=cur_val;
      atomicAdd(&output_vec[row], cur_val);
      cur_val = 0.0;
      while (j == next_row_pos) {
        ++row;
        next_row_pos = *(++next_row_idx);
      }
    }
    cur_val += val[j] * tex1Dfetch<float>(input_tex, col_idx[j]);
  }

  // output_vec[row] += cur_val;
  atomicAdd(&output_vec[row], cur_val);
}
/*
int spmv_csr_gpu_chunks(CSR csr, unsigned n, float *input_vec,
                        float *output_vec) {
  if (n != csr.ncol) {
    return 1;
  }
  CHECK_CUDA(cudaMemset(output_vec, 0, sizeof(float) * csr.nrow));
  unsigned n_data_blocks = (csr.nnz + DATA_BLOCK - 1) / DATA_BLOCK;
  unsigned nblocks = (n_data_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
  printf("nblocks %u\n", nblocks);
  printf("n_data_blocks %u\n", n_data_blocks);
  spmv_csr_gpu_kernel_chunks<<<nblocks, BLOCK_SIZE>>>(
      csr.val, csr.row_idx, csr.col_idx, input_vec, output_vec, csr.nrow,
      csr.ncol, csr.nnz);
  CHECK_CUDA(cudaDeviceSynchronize());

  return 0;
}*/
void dummy_launcher(CSR *csr, const cudaTextureObject_t input_vec,
                        float *output_vec) {
  unsigned n_data_blocks = (csr->nnz + DATA_BLOCK - 1) / DATA_BLOCK;
  unsigned nblocks = (n_data_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cudaMemset(output_vec, 0, sizeof(float) * csr->nrow);
  spmv_csr_gpu_kernel_chunks<<<nblocks, BLOCK_SIZE>>>(csr->val, csr->row_idx,
                                                      csr->col_idx, input_vec,
                                                      output_vec, csr->nrow,
                                                      csr->ncol, csr->nnz);
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

  //allocate texture
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


  TEST_FUNCTION(dummy_launcher(gpu_csr, input_tex, output_gpu));

  CHECK_CUDA(cudaDestroyTextureObject(input_tex));
  CHECK_CUDA(cudaMemcpy(output_vec, output_gpu, sizeof(float) * gpu_csr->nrow,
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(input_vec_gpu));
  CHECK_CUDA(cudaFree(output_gpu));
  free_csr_gpu(gpu_csr);
  return 0;
}

int main(int argc, char *argv[]) {
  CSR *csr = common_read_from_file(argc, argv);

  float *rand_vec; // = (float * )malloc(sizeof(float)*COLS);
  cudaMallocManaged(&rand_vec, csr->ncol * sizeof(float));
  float *output; //= (float*)malloc(sizeof(float)*COLS*(REPS+1));
  cudaMallocManaged(&output, sizeof(float) * csr->nrow * 2);
  for (unsigned i = 0; i < csr->ncol; i++) {
    rand_vec[i] = (float)(rand() % 2001 - 1000) * 0.001;
  }
  unsigned *tmp;
  cudaMallocManaged(&tmp, csr->nnz * sizeof(unsigned));
  spmv_csr_gpu_chunks(csr, csr->ncol, rand_vec, output); //, tmp
  spmv_csr(*csr, csr->ncol, rand_vec, output + csr->nrow);

  if (relative_error_compare(output, output + csr->nrow, csr->nrow)) {
    printf("Error in the output\n");
    return -1;
  }

  csr_free(csr);
  cudaFree(rand_vec);
  cudaFree(output);
  cudaFree(tmp);
  printf("test passed\n\n");
  // free(rand_vec);
  // free(output);
  // printf("test passed\n");
  return 0;
}
