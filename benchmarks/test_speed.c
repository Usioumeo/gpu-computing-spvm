
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <immintrin.h>
#define SIZE 100000000
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

float dummy_sum(float *a, float *b) {
    float val = 0;
    for(int i = 0; i < SIZE; ++i) {
        val += a[i] * b[i];
    }
    return val;
}
float simd_sum(float *a, float *b) {
    float val = 0.0f;
    __m256 sum = _mm256_setzero_ps();
    __m256 vec1, vec2, product;
    for(int i = 0; i <= SIZE; i += 8) {
        vec1 = _mm256_loadu_ps(a+=8);
        vec2 = _mm256_loadu_ps(b+=8);
        product = _mm256_mul_ps(vec1, vec2);
        sum = _mm256_add_ps(sum, product);
    }
    // Horizontal sum of the 8 floats in sum
    float temp[8];
    _mm256_storeu_ps(temp, sum);
    for(int j = 0; j < 8; ++j) {
        val += temp[j];
    }

    return val;

}
float dummy_sum_openmp(float *a, float *b) {
    float val = 0;
    #pragma omp parallel for reduction(+:val)
    for(int i = 0; i < SIZE; ++i) {
        val += a[i] * b[i];
    }
    return val;
}
float dummy_sum_openmp_simd(float *a, float *b) {
    
    __m256 sum = _mm256_setzero_ps();
    #pragma omp parallel
    {
        __m256 local_sum = _mm256_setzero_ps();
        __m256 vec1, vec2, product;
        int start = SIZE*omp_get_thread_num()/omp_get_num_threads();
        int end=SIZE*(omp_get_thread_num()+1)/omp_get_num_threads();
        //printf("%d %d\n", start, end);
        //#pragma omp for schedule(static)
        float* a1=a+start-8;
        float* b1=b+start-8;
        float* a1_end=a+end;
        /*float * a1 = a+omp_get_thread_num()*8;
        float * b1 = b+omp_get_thread_num()*8;
        float * a1_end = a+SIZE;*/
        while(a1<a1_end){
            vec1 = _mm256_loadu_ps(a1);
            vec2 = _mm256_loadu_ps(b1);
            product = _mm256_mul_ps(vec1, vec2);
            local_sum = _mm256_add_ps(local_sum, product);
            a1+=8;
            b1+=8;
        }
        #pragma omp critical
        {
            sum = _mm256_add_ps(sum, local_sum);
        }
    }
    float temp[8];
    float val = 0.0;
    _mm256_storeu_ps(temp, sum);
    for(int j = 0; j < 8; ++j) {
        val += temp[j];
    }
    return val;
}
int main(){
    omp_set_num_threads(16);
    srand(time(NULL));
    int i;
    float *a = malloc(SIZE * sizeof(float));
    float *b = malloc(SIZE * sizeof(float));
    for (i = 0; i < SIZE; i++) {
        a[i]=(rand() % 2001 - 1000)*0.001;
        b[i]=(rand() % 2001 - 1000)*0.001;
    }
    float *out = malloc(REPS * sizeof(float));
    START_TIMER
        float output = dummy_sum_openmp_simd(a, b);
        if(i>=0){
            out[i]=output;
        }
    END_TIMER
    for(int i=0; i<REPS; i++){
        printf("%f\n", out[i]);
    }
    printf("output seriale %f\n", dummy_sum(a, b));
    printf("output openmp %f\n", dummy_sum_openmp(a, b));
    printf("output simd %f\n", simd_sum(a, b));
    printf("Elapsed time: %f\n in order to do %u (avaraged on reps %u)\n", CPU_time, REPS, SIZE);

    float flops = 2.0*SIZE/CPU_time;
    printf("computed Gflops = %f\n", flops/1.0e9);
    size_t total_memory = SIZE * sizeof(float) * 2;
    float total_bandwidth = (float)total_memory / 1.0e9/CPU_time;
    printf("total memory = %f GB\n", (float)total_memory / 1.0e9);
    printf("total memory = %f GB/s\n", total_bandwidth);

    free(a);
    free(b);
    return 0;
}