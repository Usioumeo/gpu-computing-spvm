#include"../src/headers/lib.h"
#include <stdio.h>
#include <sys/time.h>
#define ROWS 65536
#define COLS 65536
#define NNZ 20000000

#define WARMUPS 10
#define REPS 100

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
    COO *coo = coo_new();
    coo_generate_random(coo, ROWS, COLS, NNZ);
    CSR *csr = csr_new();
    coo_to_csr(coo, csr);
    
    float *rand_vec = (float * )malloc(sizeof(float)*COLS);
    float *output = (float*)malloc(sizeof(float)*COLS*(REPS+1+WARMUPS));
    for(unsigned long i =0; i<COLS; i++){
        rand_vec[i]=(float)(rand()%2001-1000)*0.001;
    }
    START_TIMER
    float *out = output+i*COLS;
    if (i<0){
        out = output;
    }
    spmv_csr(*csr, COLS, rand_vec, out);
    END_TIMER
    spmv_csr(*csr, COLS, rand_vec, &output[REPS*COLS]);
    for (unsigned long j = 0; j < COLS; j++) {
        for(unsigned long i = 0; i<REPS; i++){
            if(output[i*COLS+j] != output[REPS*COLS+j]){
                printf("Error in the output\n");
                return -1;
            }
      }
    }
    
    printf("Elapsed time: %f\n in order to do %u (avaraged on reps %lu)\n", CPU_time, REPS, csr->nnz);

    float flops = 2.0*NNZ/CPU_time;
    printf("computed Gflops = %f\n", flops/1.0e9);

    
    coo_free(coo);
    csr_free(csr);
    free(rand_vec);
    free(output);
    printf("test passed\n");
    return 0;
    
}