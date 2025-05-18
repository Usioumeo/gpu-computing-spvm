//USELESS DEFINE, MAKES HAPPY THE LINTER
#define USE_OPENMP
#include"lib.h"
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#define ROWS (1<<13)
#define COLS (1<<13)
#define NNZ (1<<24)

#define WARMUPS 40
#define REPS 500

#define START_TIMER                                                            \
  struct timeval temp_1 = {0, 0}, temp_2 = {0, 0};                             \
  for (int i = -WARMUPS; i < REPS; i++) {                                      \
    if (i == 0) {                                                              \
      gettimeofday(&temp_1, NULL);                                             \
    }

#define END_TIMER                                                              \
  }                                                                            \
  gettimeofday(&temp_2, NULL);                                                 \
  float CPU_time = ((temp_2.tv_sec - temp_1.tv_sec) +                          \
                    (temp_2.tv_usec - temp_1.tv_usec) / 1e6)/REPS;

int main(){
    omp_set_num_threads(4);
    COO *coo = coo_new();
    coo_generate_random(coo, ROWS, COLS, NNZ);
    coo_sort_in_ascending_order(coo);
    CSR *csr = csr_new();
    coo_to_csr(coo, csr);
    
    float *rand_vec = (float * )malloc(sizeof(float)*COLS);
    float *output = (float*)malloc(sizeof(float)*COLS*(REPS+1));
    memset(output, 0, sizeof(float)*COLS*(REPS+1));
    for(unsigned i =0; i<COLS; i++){
        rand_vec[i]=(float)(rand()%2001-1000)*0.001;
    }
    //CSR *temp=csr_new();
  //  csr_sort_in_ascending_order(*csr);

    START_TIMER
      float *out = output+i*COLS;
      if (i<0){
          out = output;
      }
      spmv_csr_openmp_simd(*csr, COLS, rand_vec, out);
    END_TIMER

    spmv_csr(*csr, COLS, rand_vec, &output[REPS*COLS]);
    printf("Elapsed time: %f\n in order to do %u (avaraged on reps %u)\n", CPU_time, REPS, csr->nnz);

    float flops = 2.0*NNZ/CPU_time;
    printf("computed Gflops = %f\n", flops/1.0e9);

    size_t total_memory = (csr->nrow) * sizeof(unsigned) * 2 +
                        csr->nnz * (sizeof(float) + sizeof(unsigned)) +
                        csr->nnz * sizeof(float) + csr->nrow * sizeof(unsigned);

    float gbytes = (float)total_memory / 1.0e9;
    float gbytesps = gbytes / CPU_time;
    printf("total memory = %f GB\n", gbytes);
    printf("total memory = %f GB/s\n", gbytesps);

    //printf("output %lu\n", out-output);
    for (unsigned j = 0; j < COLS; j++) {
        for(unsigned i = 0; i<REPS; i++){
            if(output[i*COLS+j]- output[REPS*COLS+j]>0.1){
                printf("Error in the output %u %u %f %f %u %u\n", i, j, output[i*COLS+j], output[REPS*COLS+j], i*COLS+j, REPS*COLS+j);
                return -1;
            }
      }
    }
    
    
    

    
    coo_free(coo);
    csr_free(csr);
    free(rand_vec);
    free(output);
    printf("test passed\n");
    return 0;
    
}