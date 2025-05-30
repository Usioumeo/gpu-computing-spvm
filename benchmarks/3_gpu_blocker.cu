
#include <cassert>
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

#define WARMUPS 40
#define REPS 100

#define BLOCK_SIZE 32
#define DATA_BLOCK (16)
/*
__global__ void spmv_csr_gpu_kernel_dynamic_son(float *val, unsigned *col_idx,
                                                float *input_vec,
                                                unsigned row_size,
                                                float *output) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_size==2814){
      printf("Hello from thread %d, block %d\n", threadIdx.x, blockIdx.x);
      printf("row %u %u <= %u < %u\n", i, row_size, i,
             row_size);
      assert(1 == 0);
    }
  if (i < row_size) {
    
    atomicAdd(output, val[i] * input_vec[col_idx[i]]);
    //output+= row_val[i] * input_vec[col_idx[i]];
  }
}

__global__ void spmv_csr_gpu_kernel_dynamic(CSR csr, unsigned n,
                                            float *input_vec,
                                            float *output_vec) {
                                              __syncthreads();
  // for (unsigned i = 0; i < csr.nrow; ++i) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < csr.nrow) {

    unsigned start = csr.row_idx[i];
    unsigned end = csr.row_idx[i + 1];

    if (end - start >= 16) {
#define SON_BLOCK_SIZE 16
      unsigned n_blocks_son =
          (end - start + SON_BLOCK_SIZE - 1) / SON_BLOCK_SIZE;
      if (i < 5846347 && i > 5846147) {
        printf("row %d, length %u, n_blocks_son %u\n", i, end - start,
               n_blocks_son);
               spmv_csr_gpu_kernel_dynamic_son<<<1, end - start>>>(
          csr.val + start, csr.col_idx + start, input_vec, end - start,
          &output_vec[i]);
      }

      
    } else {
      float out = 0.0;
      float *val = csr.val + start;
      unsigned *col = csr.col_idx + start;
      float *val_end = csr.val + end;

      while (val < val_end) {

        out += *val * input_vec[*col];
        val++;
        col++;
      }
      output_vec[i] = out;
    }
  }

  //}
}

int spmv_csr_gpu_dynamic(CSR csr, unsigned n, float *input_vec,
                         float *output_vec) {
  if (n != csr.ncol) {
    return 1;
  }
  cudaMemset(output_vec, 0, sizeof(float) * csr.nrow);
  unsigned nblocks = (csr.nrow + BLOCK_SIZE - 1) / BLOCK_SIZE;
  spmv_csr_gpu_kernel_dynamic<<<nblocks, BLOCK_SIZE>>>(csr, n, input_vec,
                                                       output_vec);
  cudaDeviceSynchronize();

  return 0;
}
*/
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

__global__ void spmv_csr_gpu_kernel_chunks( const float* __restrict__ val, const unsigned * __restrict__ row_idx, const unsigned* __restrict__ col_idx,
                                           const float* __restrict__ input_vec, float *output_vec, unsigned nrow, unsigned ncol, unsigned nnz) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned starting_point = i * DATA_BLOCK;
  if (starting_point >= nnz) {
    return;
  }
  unsigned row = upper_bound(row_idx, nrow, starting_point);
  if (row_idx[row] > starting_point ||
      starting_point >= row_idx[row + 1]) {

    printf("Hello from thread %d, block %d\n", threadIdx.x, blockIdx.x);
    printf("row %u %u <= %u < %u\n", row, row_idx[row], starting_point,
           row_idx[row + 1]);
    assert(1 == 0);
  }
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
    cur_val += val[j] * input_vec[col_idx[j]];
  }

  // output_vec[row] += cur_val;
  atomicAdd(&output_vec[row], cur_val);
}


int spmv_csr_gpu_chunks(CSR csr, unsigned n, float *input_vec,
                        float *output_vec) {
  if (n != csr.ncol) {
    return 1;
  }
  /*for (int i = 0; i < csr.nrow; i++) {
    output_vec[i] = 0.0;
  }*/
  cudaMemset(output_vec, 0, sizeof(float) * csr.nrow);
  unsigned n_data_blocks = (csr.nnz + DATA_BLOCK - 1) / DATA_BLOCK;
  unsigned nblocks = (n_data_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
  /*printf("nblocks %u\n", nblocks);
  printf("n_data_blocks %u\n", n_data_blocks);*/
  spmv_csr_gpu_kernel_chunks<<<nblocks, BLOCK_SIZE>>>(
      csr.val, csr.row_idx, csr.col_idx, input_vec, output_vec, csr.nrow,
      csr.ncol, csr.nnz);
  cudaDeviceSynchronize();

  return 0;
}

__global__ void spmv_csr_gpu_kernel_nnz( const float* __restrict__ val, const unsigned * __restrict__ row_idx, const unsigned* __restrict__ col_idx,
                                           const float* __restrict__ input_vec, float *output_vec, unsigned nrow, unsigned ncol, unsigned nnz){
  // for (unsigned i = 0; i < csr.nrow; ++i) {
  unsigned start = (blockIdx.x * blockDim.x + threadIdx.x)*DATA_BLOCK;
  
  unsigned end = start + DATA_BLOCK;
  if(start>= nnz) {
    return;
  }
  if(end > nnz) {
    end = nnz;
  }
  float cur=0.0;
  unsigned prev_row = upper_bound(row_idx, nrow, start);
  for(int i=start; i<end; i++){
    unsigned cur_row = upper_bound(row_idx, nrow, i);
    if (prev_row != cur_row) {
      atomicAdd(&output_vec[prev_row], cur);
      cur = 0.0;
      prev_row = cur_row;
    }
    cur+= val[i] * input_vec[col_idx[i]];
  }
  atomicAdd(&output_vec[prev_row], cur);
}

int spmv_csr_gpu_nnz(CSR csr, unsigned n, float *input_vec, float *output_vec, unsigned* rows) {
  if (n != csr.ncol) {
    return 1;
  }
  cudaMemset(output_vec, 0, sizeof(float) * csr.nrow);
  unsigned int nblocks = (csr.nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;
  /*set_rows<<<nblocks, BLOCK_SIZE>>>(csr, rows);
  cudaDeviceSynchronize();*/
  unsigned n_data_blocks = (csr.nnz + DATA_BLOCK - 1) / DATA_BLOCK;
  nblocks = (n_data_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
  spmv_csr_gpu_kernel_nnz<<<nblocks, BLOCK_SIZE>>>(
      csr.val, csr.row_idx, csr.col_idx, input_vec, output_vec, csr.nrow,
      csr.ncol, csr.nnz);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  return 0;
}

int main(int argc, char *argv[]) {
  COO *coo = coo_new();
  if(argc > 2) {
    printf("Usage: %s <input_file>\n", argv[0]);
    return -1;
  }
  if (argc==2) {
    FILE *input = fopen(argv[1], "r");
    if (input == NULL) {
      printf("Error opening file: %s\n", argv[1]);
      return -1;
    }
    if (coo_from_file(input, coo)!=0){
      printf("Error reading COO from file: %s\n", argv[1]);
      fclose(input);
      return -1;
    }
  } else{
    coo_generate_random(coo, ROWS, COLS, NNZ);
  }
  CSR *csr = csr_new();
  coo_to_csr(coo, csr);
  printf("coo->nrow %u coo->ncol %u coo->nnz %u\n", coo->nrow, coo->ncol,
         coo->nnz);
  float *rand_vec; // = (float * )malloc(sizeof(float)*COLS);
  cudaMallocManaged(&rand_vec, csr->ncol * sizeof(float));
  float *output; //= (float*)malloc(sizeof(float)*COLS*(REPS+1));
  cudaMallocManaged(&output, sizeof(float) * csr->nrow * 2);
  for (unsigned i = 0; i < csr->ncol; i++) {
    rand_vec[i] = (float)(rand() % 2001 - 1000) * 0.001;
  }
  unsigned * tmp;
  cudaMallocManaged(&tmp, csr->nnz * sizeof(unsigned));
  TEST_FUNCTION(spmv_csr_gpu_chunks(*csr, csr->ncol, rand_vec, output)); //, tmp
  spmv_csr(*csr,  csr->ncol, rand_vec, output +  csr->nrow);

  if (relative_error_compare(output, output + csr->nrow, csr->nrow)) {
    printf("Error in the output\n");
    return -1;
  }

  coo_free(coo);
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
