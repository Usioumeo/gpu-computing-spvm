#ifndef LIBCSR_H
#define LIBCSR_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

#define TEST_FUNCTION(func) \
        START_TIMER \
        func; \
        END_TIMER \
  printf("Elapsed time: %f\n", CPU_time); \
  printf("Total reps: %u\n", REPS); \
  printf("Nrow: %u\n", csr->nrow); \
  printf("Ncol: %u\n", csr->ncol); \
  printf("Nnz: %u\n", csr->nnz); \
  float flops = 2.0 * csr->nnz / CPU_time; \
  printf("Gflops: %f\n", flops / 1.0e9);\
  size_t total_memory = (csr->nrow) * sizeof(unsigned) * 2 + \
                        csr->nnz * (sizeof(float) + sizeof(unsigned)) +\
                        csr->nnz * sizeof(float) + csr->nrow * sizeof(float);\
  float gbytes = (float)total_memory / 1.0e9;\
  float gbytesps = gbytes / CPU_time;\
  printf("Total memory moved: %f GB\n", gbytes);\
  printf("Total bandwidth: %f GB/s\n", gbytesps); \

typedef struct {
    // row index (0 indexed)
    unsigned row;
    // col index (0 indexed)
    unsigned col;
    // value contained
    float val; 
} COOEntry;

/*
COO Matrix
*/
typedef struct {
    // multiple COO entries
    COOEntry *data;
    //non-zero elements, aka number of COO entries,
    unsigned nnz;
    // how many rows
    unsigned nrow;
    // how many columns
    unsigned ncol;
}COO;


/*
CSR Matrix
to obtain the number of non-
*/
typedef struct{
    // one row index for each row +1 (and value refers to col_idx and val)
    unsigned *row_idx;
    // array of columns, there is one for each non-zero element
    unsigned *col_idx;
    // array of values, there is one for each non-zero element
    float *val;
    // how many rows
    unsigned nrow;
    // how many columns
    unsigned ncol;
    // how many non-zero elements (its not strictly necessary to store it, it can be computed as row_idx[nrow])
    unsigned nnz;
}CSR;

// Function to generate random sparse matrix in COO format
void coo_generate_random(COO *coo, unsigned rows, unsigned cols, unsigned nnz);

// Function to convert COO to CSR
// 
// assumes that the csr matrix is already allocated and big enough
// it doesn't destroy the orifinal coo matrix
void coo_to_csr(COO *coo, CSR *csr);

// Function to convert CSR to COO
// it doesn't destroy the orifinal csr matrix
void csr_to_coo(CSR *csr, COO *coo);


// compare function for qsort
int cooEntry_compare(const void *a, const void *b);

// Function to sort COO entries by row and column indices
void coo_sort_in_ascending_order(COO *coo);


// Function to free the memory allocated for COO matrix
// IT DOES NOT FREE THE POINTER, JUST THE DATA
void coo_free(COO *coo);

// Function to free the memory allocated for CSR matrix
// IT DOES NOT FREE THE POINTER, JUST THE DATA
void csr_free(CSR *csr);


// Function to free the memory allocated for CSR matrix
// IT ALSO FREES THE POINTER
void csr_free(CSR *csr);

// Function to create a new empty COO matrix
COO* coo_new();

// Function to create a new empty CSR matrix
CSR* csr_new();

void coo_reserve(COO *coo, unsigned nnz);

int coo_from_file(FILE *input, COO *coo);


// compare function for qsort
int compare_cooEntry(const void *a, const void *b);

int coo_write_to_file(FILE *output, COO *coo);

int coo_compare(COO *coo1, COO *coo2);
// Function to reserve memory for CSR matrix
void csr_reserve(CSR *csr, unsigned nnz, unsigned nrow);

int spmv_coo(COO coo, unsigned n, float *input_vec, float * output_vec);

// default implementation, it should be the correct version
int spmv_csr(CSR csr, unsigned n, float *input_vec, float * output_vec);

int write_bin_to_file(CSR *csr, const char *filename);
int read_bin_to_csr(const char *filename, CSR *csr);
int spmv_csr_block(CSR csr, unsigned n, float *input_vec, float * output_vec);

CSR *copy_csr_to_gpu(CSR *csr);

void free_csr_gpu(CSR *csr);

#define CHECK_CUDA(call)                                                       \
  if ((call) != cudaSuccess) {                                                 \
    fprintf(stderr, "CUDA error at %s:%u %u\n", __FILE__, __LINE__, call);     \
    exit(1);                                                                   \
  }


#ifdef USE_OPENMP
int spmv_csr_openmp(CSR csr, unsigned n, float *input_vec, float * output_vec);
int spmv_csr_openmp_simd(CSR csr, unsigned n, float *input_vec, float * output_vec);
#endif
int spmv_csr_sort(CSR csr, unsigned n, float *input_vec, float * output_vec);
void csr_sort_in_ascending_order(CSR csr);
int relative_error_compare(float *a, float *b, unsigned n);

#endif