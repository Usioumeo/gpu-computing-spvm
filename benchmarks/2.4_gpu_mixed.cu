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
#define V1_BLOCK_SIZE 32

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
                                        cudaTextureObject_t input_tex,
                                        float *output_vec, unsigned nrow,
                                        unsigned ncol, unsigned nnz) {
  __shared__ unsigned shared_rows_idx[DATA_BLOCK];
  __shared__ float contributions[DATA_BLOCK];

  unsigned block_start = blockIdx.x * DATA_BLOCK;
  unsigned block_end = min(block_start + DATA_BLOCK, nnz);
  /// build the shared memory with the row_idx
  unsigned assigned_start =
      block_start + DATA_BLOCK * threadIdx.x / BLOCK_THREADS;
  unsigned assigned_end = min(
      block_start + DATA_BLOCK * (threadIdx.x + 1) / BLOCK_THREADS, block_end);

  unsigned row = 0;
  for (unsigned i = assigned_start; i < assigned_end;) {
    row = normal_upper_bound(row_idx, nrow, i);

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
      contributions[i - block_start] = val[i] * tex1Dfetch<float>(input_tex, col_idx[i]);
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

__global__ void measure_sparsity(const unsigned *__restrict__ row_idx,
                                unsigned nrow, 
                                double *stats) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < nrow) {
        // Calculate row length
        double row_length = (double)(row_idx[tid + 1] - row_idx[tid]);
        
        // Use atomic operations to compute statistics
        atomicAdd(stats, row_length); // sum
        atomicAdd(stats+1, (row_length * row_length)); // sum of squares
    }
}
__global__ void measure_sparsity_optimized(const unsigned *__restrict__ row_idx,
                                         unsigned nrow, 
                                         double *stats, unsigned compute1_every =1) {
    __shared__ double sums[32];
    __shared__ double sq_sums[32];
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned local_tid = threadIdx.x;
    unsigned lane = threadIdx.x % 32; // Warp lane index
    unsigned warp_id = threadIdx.x / 32;
    // Initialize shared memory
    sums[local_tid] = 0.0f;
    sq_sums[local_tid] = 0.0f;
    
    // Each thread processes multiple elements to improve memory coalescing
    double local_sum = 0.0f;
    double local_sum_sq = 0.0f;
    __syncwarp();
    //#pragma unroll 4
    for (unsigned i = tid; i < nrow; i += gridDim.x * blockDim.x*compute1_every) {
        double row_length = (double)(row_idx[i + 1] - row_idx[i]);
        local_sum += row_length;
        local_sum_sq += (row_length * row_length);
    }

    __syncwarp();
    //#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2){
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }
    // Store results in shared memory
    if(lane==0){
        sums[warp_id] = local_sum;
        sq_sums[warp_id] = local_sum_sq;
    }

    __syncthreads();
    if (warp_id == 0) {
        unsigned num_warps = (blockDim.x + 31) / 32;
        local_sum = (lane < num_warps) ? sums[lane] : 0.0f;
        local_sum_sq = (lane < num_warps) ? sq_sums[lane] : 0.0f;
        
        //#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
            local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
        }
        
        // Only first thread in block performs atomic operations
        if (threadIdx.x == 0) {
            atomicAdd(&stats[0], local_sum);
            atomicAdd(&stats[1], local_sum_sq);
        }
    }
}

double* status_gpu=NULL;

// Add this function to analyze workload balance
float analyze_workload_balance(CSR *gpu_csr) {
    if(status_gpu==NULL){
        cudaMalloc(&status_gpu, sizeof(double) * 2);
    }
    cudaMemset(status_gpu, 0, sizeof(double)*2);
    double stats_host[2]; // sum, sum_sq


    // Launch kernel to measure sparsity
    
    unsigned block_size = 256;
    unsigned grid_size = std::min(8192u, (gpu_csr->nrow + block_size - 1) / block_size);
    
    measure_sparsity_optimized<<<grid_size, block_size, 2 * block_size * sizeof(double)>>>(
        gpu_csr->row_idx, gpu_csr->nrow, status_gpu, 10);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Copy results back
    CHECK_CUDA(cudaMemcpy(stats_host, status_gpu, sizeof(double) * 2, cudaMemcpyDeviceToHost));
    // Calculate coefficient of variation
    double mean = stats_host[0] / gpu_csr->nrow;
    double variance = (stats_host[1] / gpu_csr->nrow) - (mean * mean);
    double std_dev = sqrt(variance);
    double cv = std_dev / mean; // Coefficient of variation
    
    /*printf("Workload Analysis:\n");
    printf("  Average row length: %.2f\n", mean);
    printf("  Standard deviation: %.2f\n", std_dev);
    printf("  Coefficient of variation: %.3f\n", cv);*/
    
    
    return cv;
}

__global__ void spmv_csr_gpu_kernel_texture(CSR csr, unsigned n,
                                            cudaTextureObject_t input_tex,
                                            float *__restrict__ output_vec) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < csr.nrow) {
    float out = 0.0;
    unsigned start = csr.row_idx[i];
    unsigned end = csr.row_idx[i + 1];
    unsigned idx = start;

    // compute
    while (idx < end) {
      unsigned col = __ldg(csr.col_idx + idx);
      float input_val = tex1Dfetch<float>(input_tex, col);
      float val = __ldg(csr.val + idx);
      // Use texture memory to fetch the input vector

      out += val * input_val;
      idx++;
    }

    output_vec[i] = out;
  }
}

// Add texture memory launcher
inline void dummy_launcher_v1(CSR *csr, cudaTextureObject_t input_tex, float *output_vec) {
  unsigned nblocks = (csr->nrow + V1_BLOCK_SIZE - 1) / V1_BLOCK_SIZE;

  // Launch kernel with texture
  spmv_csr_gpu_kernel_texture<<<nblocks, V1_BLOCK_SIZE>>>(*csr, csr->ncol,
                                                       input_tex, output_vec);
  CHECK_CUDA(cudaDeviceSynchronize());

  
}

inline void dummy_launcher_v2(CSR *csr, cudaTextureObject_t input_tex, float *output_vec) {
    cudaMemset(output_vec, 0, sizeof(float) * csr->nrow);
    unsigned int nblocks = (csr->nnz + DATA_BLOCK - 1) / DATA_BLOCK;
    spmv_csr_gpu_kernel_nnz<<<nblocks, BLOCK_THREADS>>>(
        csr->val, csr->row_idx, csr->col_idx, input_tex, output_vec, csr->nrow,
        csr->ncol, csr->nnz);
  
  CHECK_CUDA(cudaDeviceSynchronize());
}
inline void dummy_launcher(CSR *csr, cudaTextureObject_t input_tex, float *output_vec) {
  if (analyze_workload_balance(csr)>10.0){
    dummy_launcher_v2(csr, input_tex, output_vec);
  }else{
    dummy_launcher_v1(csr, input_tex, output_vec);
  }
  
  //CHECK_CUDA(cudaDeviceSynchronize());
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
