
/*
Entry to COO
*/

#include "headers/lib.h"
#include "headers/mmio.h"
#include "headers/uthash.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct {
  UT_hash_handle hh;
 uint64_t key;
} CooDedup;

uint64_t key_gen(unsigned long row, unsigned long col) {
  uint64_t key = 0;
  key |= (uint64_t)row << 32;
  key |= (uint64_t)col;
  return key;
}
// Function to generate random sparse matrix in COO format
void coo_generate_random(COO *coo, unsigned long rows, unsigned long cols,
                         unsigned long nnz) {
if (nnz > rows * cols) {
    printf("Error: nnz cannot be greater than rows * cols\n");
    exit(1);
  }
  CooDedup *set = NULL, *entry;
  CooDedup *tmp_set = (CooDedup*)malloc(nnz * sizeof(CooDedup)); 
  coo->nrow = rows;
  coo->ncol = cols;
  coo_reserve(coo, nnz);

  for (unsigned long i = 0; i < coo->nnz; ++i) {
    coo->data[i].row = (unsigned long)rand() % rows;
    coo->data[i].col = (unsigned long)rand() % cols;
    uint64_t key = key_gen(coo->data[i].row, coo->data[i].col);
    CooDedup *entry;
    HASH_FIND(hh, set, &key, sizeof(uint64_t), entry);
    if(entry){
      // Duplicate found, decrement nnz and continue
      i--;
      continue;
    }
    tmp_set[i].key = key;
    entry = &tmp_set[i];
    HASH_ADD(hh, set, key, sizeof(uint64_t), entry);    
    coo->data[i].val = (float)(rand() % 2001 - 1000) / 1000.0f; // Random value
  }
  HASH_CLEAR(hh, set);
  free(tmp_set);
}

// Function to reserve memory for COO matrix
void coo_reserve(COO *coo, unsigned long nnz) {
  coo->data = (COOEntry *)realloc(coo->data, nnz * sizeof(COOEntry));
  coo->nnz = nnz;
}
// Function to reserve memory for CSR matrix
void csr_reserve(CSR *csr, unsigned long nnz, unsigned long nrow) {
  csr->nnz = nnz;
  csr->nrow = nrow;
  // resize csr arrays
  csr->row_idx = (unsigned long *)realloc(
      csr->row_idx, (nrow + 1) * sizeof(unsigned long));
  csr->col_idx =
      (unsigned long *)realloc(csr->col_idx, nnz * sizeof(unsigned long));
  csr->val = (float *)realloc(csr->val, nnz * sizeof(float));
}


// Function to convert COO to CSR
//
// assumes that the csr matrix is already allocated and big enough
// it doesn't destroy the orifinal coo matrix
void coo_to_csr(COO *coo, CSR *csr) {
  csr->ncol = coo->ncol;
  // resize csr arrays
  csr_reserve(csr, coo->nnz, coo->nrow);

  // initialize row_idx to 0
  for (unsigned long i = 0; i <= csr->nrow; ++i)
    csr->row_idx[i] = 0;

  // count the number of non-zero elements in each row
  for (unsigned long i = 0; i < coo->nnz; ++i)
    csr->row_idx[coo->data[i].row + 1]++;

  // compute the prefix sum to get the row pointers
  for (unsigned long i = 0; i < csr->nrow; ++i)
    csr->row_idx[i + 1] += csr->row_idx[i];

  // temporary array to keep track of the current index in each row
  unsigned long *temp =
      (unsigned long *)malloc(csr->nrow * sizeof(unsigned long));

  // initialize temp to the row pointers
  for (unsigned long i = 0; i < csr->nrow; ++i)
    temp[i] = csr->row_idx[i];

  // fill the col_idx and val arrays
  for (unsigned long i = 0; i < csr->nnz; ++i) {
    unsigned long row = coo->data[i].row;
    unsigned long idx = temp[row]++;
    csr->col_idx[idx] = coo->data[i].col;
    csr->val[idx] = coo->data[i].val;
  }
  // free the temporary array
  free(temp);
}

// Function to convert CSR to COO

// it doesn't destroy the orifinal csr matrix
void csr_to_coo(CSR *csr, COO *coo) {
  // init coo
  coo->nrow = csr->nrow;
  coo->ncol = csr->ncol;
  coo_reserve(coo, csr->nnz);

  for (unsigned long i = 0; i < csr->nrow; ++i) {
    // for each row
    for (unsigned long j = csr->row_idx[i]; j < csr->row_idx[i + 1]; ++j) {
      // for each entry in the row
      coo->data[j].row = i;
      coo->data[j].col = csr->col_idx[j];
      coo->data[j].val = csr->val[j];
    }
  }
}

// compare function for qsort
int compare_cooEntry(const void *a, const void *b) {
  COOEntry *entryA = (COOEntry *)a;
  COOEntry *entryB = (COOEntry *)b;

  if (entryA->col != entryB->col) {
    return (int)entryA->col - (int)entryB->col;
  } else {
    return (int)entryA->row - (int)entryB->row;
  }
}

// Function to sort COO entries by row and column indices
void coo_sort_in_ascending_order(COO *coo) {
  qsort(coo->data, (long unsigned)coo->nnz, sizeof(COOEntry), compare_cooEntry);
}

// Function to free the memory allocated for COO matrix
// IT ALSO FREES THE POINTER
void coo_free(COO *coo) {
  free(coo->data);
  // coo->data = NULL;
  free(coo);
}

// Function to free the memory allocated for CSR matrix
// IT ALSO FREES THE POINTER
void csr_free(CSR *csr) {
  free(csr->row_idx);
  free(csr->col_idx);
  free(csr->val);
  /*csr->row_idx = NULL;
  csr->col_idx = NULL;
  csr->val = NULL;*/
  free(csr);
}

// Function to create a new empty COO matrix
COO *coo_new() {
  COO *coo = (COO *)malloc(sizeof(COO));
  coo->data = NULL;
  coo->nnz = 0;
  coo->nrow = 0;
  coo->ncol = 0;
  return coo;
}

// Function to create a new empty CSR matrix
CSR *csr_new() {
  CSR *csr = (CSR *)malloc(sizeof(CSR));
  csr->row_idx = NULL;
  csr->col_idx = NULL;
  csr->val = NULL;
  csr->nnz = 0;
  csr->nrow = 0;
  csr->ncol = 0;
  return csr;
}

int coo_from_file(FILE *input, COO *coo) {
  MM_typecode matcode;

  if (mm_read_banner(input, &matcode) != 0) {
    printf("Could not process Matrix Market banner.\n");
    exit(1);
  }

  /*  This is how one can screen matrix types if their application */
  /*  only supports a subset of the Matrix Market data types.      */

  if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode)) { //&&
    printf("Sorry, this application does not support ");
    printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    exit(1);
  }

  /* find out size of sparse matrix .... */
  int M, N, nnz;
  if (mm_read_mtx_crd_size(input, &M, &N, &nnz) != 0)
    exit(1);
  coo->nrow = M;
  coo->ncol = N;
  coo->nnz = nnz;

  /* reseve memory for matrices */
  coo_reserve(coo, coo->nnz);

  /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
  /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
  /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

  for (unsigned int i = 0; i < coo->nnz; i++) {
    fscanf(input, "%lu %lu %f\n", &coo->data[i].row, &coo->data[i].col,
           &coo->data[i].val);
    coo->data[i].row--; /* adjust from 1-based to 0-based */
    coo->data[i].col--;
  }

  if (input != stdin)
    fclose(input);
  return 0;
}

int coo_write_to_file(FILE *output, COO *coo) {
  MM_typecode matcode;
  unsigned long i;

  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_coordinate(&matcode);
  mm_set_real(&matcode);

  mm_write_banner(output, matcode);
  mm_write_mtx_crd_size(output, coo->nrow, coo->ncol, coo->nnz);

  /* NOTE: matrix market files use 1-based indices, i.e. first element
    of a vector has index 1, not 0.  */

  for (i = 0; i < coo->nnz; i++)
    fprintf(output, "%ld %ld %10.3f\n", coo->data[i].row + 1,
            coo->data[i].col + 1, coo->data[i].val);
  fclose(output);
  return 0;
}

int coo_compare(COO *coo1, COO *coo2) {
  if (coo1->nnz != coo2->nnz || coo1->nrow != coo2->nrow ||
      coo1->ncol != coo2->ncol) {
    printf("coo1 %ld %ld %ld\n", coo1->nrow, coo1->ncol, coo1->nnz);
    printf("coo2 %ld %ld %ld\n", coo2->nrow, coo2->ncol, coo2->nnz);
    return 1;
  }
  coo_sort_in_ascending_order(coo1);
  coo_sort_in_ascending_order(coo2);

  for (unsigned long i = 0; i < coo1->nnz; ++i) {
    if (coo1->data[i].row != coo2->data[i].row ||
        coo1->data[i].col != coo2->data[i].col ||
        coo1->data[i].val - coo2->data[i].val > 0.00000001) {
      printf("coo1 %ld %ld %f\n", coo1->data[i].row, coo1->data[i].col,
             coo1->data[i].val);
      printf("coo2 %ld %ld %f\n", coo2->data[i].row, coo2->data[i].col,
             coo2->data[i].val);
      return 1;
    }
  }
  return 0;
}

int spmv_coo(COO coo, unsigned long n, float *input_vec, float *output_vec) {
  if (n != coo.ncol) {
    return 1;
  }
  for(int i=0; i<coo.nrow; i++){
    output_vec[i] = 0.0;
  }
  for (unsigned long i = 0; i < coo.nnz; ++i) {
    output_vec[coo.data[i].row] += coo.data[i].val * input_vec[coo.data[i].col];
  }
  return 0;
}



// default implementation, it should be the correct version
int spmv_csr(CSR csr, unsigned long n, float *input_vec, float * output_vec){
  if (n!=csr.ncol){
    return 1;
  }
  for (unsigned long i = 0; i < csr.nrow; ++i) {
    output_vec[i] = 0.0;
      for (unsigned long j = csr.row_idx[i]; j < csr.row_idx[i + 1]; ++j) {
          output_vec[i] += csr.val[j]* input_vec[csr.col_idx[j]];
      }
  }
  
  return 0;
}
#define BLOCK_SIZE 8
int spmv_csr_block(CSR csr, unsigned long n, float *input_vec, float * output_vec){
  if (n!=csr.ncol){
    return 1;
  }
  for (unsigned long block_start = 0; block_start < csr.nrow; block_start += BLOCK_SIZE) {
    unsigned long block_max = (block_start + BLOCK_SIZE < csr.nrow) ? (block_start + BLOCK_SIZE) : csr.nrow;
    for (unsigned long i = block_start; i < block_max; ++i) {
      output_vec[i] = 0.0;

      for (unsigned long j = csr.row_idx[i]; j < csr.row_idx[i + 1]; ++j) {
          output_vec[i] += csr.val[j]* input_vec[csr.col_idx[j]];
      }
    }
  }
  
  return 0;
}

typedef struct {
  float val;
  unsigned long col;
} TmpStruct;

int compare_tmpStruct(const void *a, const void *b) {
  TmpStruct *entryA = (TmpStruct *)a;
  TmpStruct *entryB = (TmpStruct *)b;


  return (int)entryA->col - (int)entryB->col;
  
}

int spmv_csr_sort(CSR csr, unsigned long n, float *input_vec, float * output_vec){
  if (n!=csr.ncol){
    return 1;
  }
  TmpStruct *tmp = (TmpStruct*)malloc(csr.nrow*sizeof(TmpStruct));
  for (unsigned long i = 0; i < csr.nrow; ++i) {
    output_vec[i] = 0.0;
    unsigned int start = csr.row_idx[i];
    unsigned int end = csr.row_idx[i + 1];
    for(unsigned long j = start; j < end; ++j) {
      tmp[j-start].val = csr.val[j];
      tmp[j-start].col = csr.col_idx[j];
    }
    qsort(tmp, end -start, sizeof(TmpStruct), compare_tmpStruct);
    for (unsigned long j = start; j < end; ++j) {
        output_vec[i] += tmp[j-start].val* input_vec[tmp[j-start].col];
    }
  }
  free(tmp);
  
  return 0;
}
void csr_sort_in_ascending_order(CSR csr) {
  TmpStruct *tmp = (TmpStruct*)malloc(csr.nrow*sizeof(TmpStruct));
  for (unsigned long i = 0; i < csr.nrow; ++i) {
    unsigned int start = csr.row_idx[i];
    unsigned int end = csr.row_idx[i + 1];
    for(unsigned long j = start; j < end; ++j) {
      tmp[j-start].val = csr.val[j];
      tmp[j-start].col = csr.col_idx[j];
    }
    qsort(tmp, end -start, sizeof(TmpStruct), compare_tmpStruct);
    for (unsigned long j = start; j < end; ++j) {
        csr.val[j] = tmp[j-start].val;
        csr.col_idx[j] = tmp[j-start].col;
    }
  }
}

int spmv_csr_openmp(CSR csr, unsigned long n, float *input_vec, float * output_vec){
  if (n!=csr.ncol){
    return 1;
  }
  #pragma omp parallel for
  for (unsigned long i = 0; i < csr.nrow; ++i) {
      output_vec[i] = 0;
      // no race conditions, because each thread writes on a different index of the output vector
      #pragma omp simd 
      for (unsigned long j = csr.row_idx[i]; j < csr.row_idx[i + 1]; ++j) {
          output_vec[i] += csr.val[j]* input_vec[csr.col_idx[j]];
      }
  }
  return 0;
}
int spmv_csr_order(CSR csr, unsigned long n, float *input_vec, float * output_vec){
  if (n!=csr.ncol){
    return 1;
  }
  #pragma omp parallel for
  for (unsigned long i = 0; i < csr.nrow; ++i) {
      output_vec[i] = 0;
      // no race conditions, because each thread writes on a different index of the output vector
      #pragma omp simd 
      for (unsigned long j = csr.row_idx[i]; j < csr.row_idx[i + 1]; ++j) {
          output_vec[i] += csr.val[j]* input_vec[csr.col_idx[j]];
      }
  }
  return 0;
}

//int* out_row_ptr, int* out_col_ind, double* out_values
void csr_transpose(
  CSR input, CSR* output
    
) {
    csr_reserve(output, input.nnz, input.ncol);
    output->nrow = input.ncol;
    // Step 1: Count number of entries per column (which become rows in the transpose)
    int* col_counts = malloc(input.ncol*sizeof(int));
    for (int i = 0; i < input.nnz; i++) {
        col_counts[input.col_idx[i]]++;
    }

    // Step 2: Compute row_ptr for transpose
    output->row_idx[0] = 0;
    for (int i = 0; i < input.ncol; i++) {
        output->row_idx[i + 1] = output->row_idx[i] + col_counts[i];
    }

    // Temporary working array to keep track of fill-in position
    int* current_position = malloc(input.ncol * sizeof(int));
    memcpy(current_position, output->row_idx, input.ncol * sizeof(int));

    // Step 3: Fill in col_ind and values for the transpose
    for (int row = 0; row < input.nrow; row++) {
        for (int idx = input.row_idx[row]; idx < input.row_idx[row + 1]; idx++) {
            int col = input.col_idx[idx];
            int dest_pos = current_position[col];

            output->col_idx[dest_pos] = row;        // transpose: col becomes row
            output->val[dest_pos] = input.val[idx];

            current_position[col]++;
        }
    }

    free(col_counts);
    free(current_position);
}

int spmv_csr_transpose(CSR csr, CSR transpose, unsigned long n, float *input_vec, float * output_vec){
  if (n!=csr.ncol){
    return 1;
  }
  csr_transpose(csr, &transpose);

  /*for (unsigned long i = 0; i < csr.nrow; ++i) {
      for (unsigned long j = csr.row_idx[i]; j < csr.row_idx[i + 1]; ++j) {
        //printf("%lu\n", j);
          output_vec[i] += csr.val[j]* input_vec[csr.col_idx[j]];
      }
  }*/
  return 0;
}