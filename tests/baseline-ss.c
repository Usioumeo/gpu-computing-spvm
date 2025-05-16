#include"../src/headers/lib.h"
#include <stdio.h>
#include <cs.h>
#include <sys/time.h>
#define ROWS 4194304
#define COLS 4194304
#define NNZ 20000000

#define WARMUPS 4
#define REPS 10

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
    //CSR *csr = csr_new();
    //coo_to_csr(coo, csr);
    
    cs *A = cs_spalloc(ROWS, COLS, NNZ, 1, 0);
    for(unsigned long i = 0; i<coo->nnz; i++){
        cs_entry(A, coo->data[i].row, coo->data[i].col, coo->data[i].val);
    }

    // Compress to CSC
    cs *C = cs_compress(A);
    cs_spfree(A);


    
    double *rand_vec = (double * )malloc(sizeof(double)*COLS);
    double *output = (double*)malloc(sizeof(double)*COLS*(REPS+1));
    for(unsigned long i =0; i<COLS; i++){
        rand_vec[i]=(double)(rand()%2001-1000)*0.001;
    }
    //A = cs_di_spalloc(n, n, 4, 1, 1);
    START_TIMER
    cs_gaxpy(C, rand_vec, &output[i*COLS]);
    END_TIMER
    printf("Elapsed time: %f\n in order to do %u reps of %d\n", CPU_time, REPS, NNZ);

    float flops = 2.0*NNZ/CPU_time;
    printf("computed Gflops = %f\n", flops/1.0e9);

    
    coo_free(coo);
    //csr_free(csr);
    free(rand_vec);
    free(output);
    printf("test passed\n");
    return 0;
    
}