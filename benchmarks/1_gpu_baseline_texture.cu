
#include <cassert>
extern "C" {
#include "lib.h"
}
#include <math.h>
#include <stdio.h>
#include <sys/select.h>
#include <sys/time.h>

#include <stdint.h>  


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
        
        
        // compute
        while (idx < end) {
          unsigned col = __ldg(csr.col_idx +idx);
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
  CSR *csr = common_read_from_file(argc, argv);

  float *input = common_generate_random_input(csr);

  float *output = (float *)malloc(sizeof(float) * csr->nrow * 2);

  spmv_csr_gpu(csr, csr->ncol, input, output); //, tmp
  spmv_csr(*csr, csr->ncol, input, output + csr->nrow);

  if (relative_error_compare(output, output + csr->nrow, csr->nrow)) {
    printf("Error in the output\n");
    return -1;
  }

  csr_free(csr);
  free(input);
  free(output);
  printf("test passed\n\n");
  return 0;
}