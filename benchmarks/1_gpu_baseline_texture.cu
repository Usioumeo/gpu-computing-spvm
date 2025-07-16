
#include <cassert>
extern "C" {
#define USE_CUDA
#include "lib.h"
}
#include <math.h>
#include <stdio.h>
#include <sys/select.h>
#include <sys/time.h>

#include <stdint.h>  
#define ROWS (1 << 13)
#define COLS (1 << 13)
#define NNZ (1 << 24)

#define WARMUPS 4
#define REPS 10

#define BLOCK_SIZE 32
 

__global__ void spmv_csr_gpu_kernel_texture(CSR csr, unsigned n, 
                                                        cudaTextureObject_t input_tex,
                                                        float *__restrict__ output_vec) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < csr.nrow) {
        float out = 0.0;
        unsigned start = csr.row_idx[i];
        unsigned end = csr.row_idx[i + 1];
        unsigned idx = start;
        
        // Process 4 elements at once when possible AND aligned
        while (idx + 3 < end && (idx % 4 == 0)) {
            // Check if we can safely load float4 (16-byte aligned)
            if (((uintptr_t)(&csr.val[idx]) % 16 == 0) && 
                ((uintptr_t)(&csr.col_idx[idx]) % 16 == 0)) {
                
                float4 val4 = *reinterpret_cast<const float4*>(&csr.val[idx]);
                uint4 col4 = *reinterpret_cast<const uint4*>(&csr.col_idx[idx]);
                
                // Sequential texture fetches instead of concurrent
                float input0 = tex1Dfetch<float>(input_tex, col4.x);
                float input1 = tex1Dfetch<float>(input_tex, col4.y);
                float input2 = tex1Dfetch<float>(input_tex, col4.z);
                float input3 = tex1Dfetch<float>(input_tex, col4.w);
                
                out += val4.x * input0 + val4.y * input1 + val4.z * input2 + val4.w * input3;
                idx += 4;
            } else {
                // Fall back to scalar access
                out += csr.val[idx] * tex1Dfetch<float>(input_tex, csr.col_idx[idx]);
                idx++;
            }
        }
        
        // Handle remaining elements
        while (idx < end) {
            out += csr.val[idx] * tex1Dfetch<float>(input_tex, csr.col_idx[idx]);
            idx++;
        }
        
        output_vec[i] = out;
    }
}


// Add texture memory launcher
void dummy_launcher_texture(CSR *csr, float *input_vec, float *output_vec) {
    unsigned nblocks = (csr->nrow + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Create texture object
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = input_vec;
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
    
    // Launch kernel with texture
    spmv_csr_gpu_kernel_texture<<<nblocks, BLOCK_SIZE>>>(*csr, csr->ncol, input_tex, output_vec);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Clean up texture object
    CHECK_CUDA(cudaDestroyTextureObject(input_tex));
}


int spmv_csr_gpu(CSR *csr, unsigned n, float *input_vec,
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

  TEST_FUNCTION(dummy_launcher_texture(gpu_csr, input_vec_gpu, output_gpu));

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

  spmv_csr_gpu(csr, csr->ncol, input, output); //, tmp
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