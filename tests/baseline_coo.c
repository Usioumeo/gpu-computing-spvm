#include"../src/headers/lib.h"
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#define ROWS (1<<13)
#define COLS (1<<13)
#define NNZ (1<<24)

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
    coo_sort_in_ascending_order(coo);
    
    float *rand_vec = (float * )malloc(sizeof(float)*COLS);
    float *output = (float*)malloc(sizeof(float)*COLS*(REPS+1));
    memset(output, 0, sizeof(float)*COLS*(REPS+1));
    for(unsigned long i =0; i<COLS; i++){
        rand_vec[i]=(float)(rand()%2001-1000)*0.001;
    }
    //CSR *temp=csr_new();
  //  csr_sort_in_ascending_order(*csr);

    START_TIMER
      float *out = output+i*COLS;
      if (i<0){
          out = output;
      }
      spmv_coo(*coo, COLS, rand_vec, out);
    END_TIMER

    spmv_coo(*coo, COLS, rand_vec, &output[REPS*COLS]);
    printf("Elapsed time: %f\n in order to do %u (avaraged on reps %lu)\n", CPU_time, REPS, coo->nnz);

    float flops = 2.0*NNZ/CPU_time;
    printf("computed Gflops = %f\n", flops/1.0e9);

    //printf("output %lu\n", out-output);
    for (unsigned long j = 0; j < COLS; j++) {
        for(unsigned long i = 0; i<REPS; i++){
            if(output[i*COLS+j]- output[REPS*COLS+j]>0.00001){
                printf("Error in the output %lu %lu %f %f %lu %lu\n", i, j, output[i*COLS+j], output[REPS*COLS+j], i*COLS+j, REPS*COLS+j);
                return -1;
            }
      }
    }
    
    

    
    coo_free(coo);
    free(rand_vec);
    free(output);
    printf("test passed\n");
    return 0;
    
}