
/*
Entry to COO
*/

#include "headers/lib.h"
#include "headers/mmio.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to generate random sparse matrix in COO format
void coo_generate_random(COO *coo, unsigned long rows, unsigned long cols,
                         unsigned long nnz) {
  coo->nrow = rows;
  coo->ncol = cols;
  coo_reserve(coo, nnz);
  for (unsigned long i = 0; i < coo->nnz; ++i) {
    coo->data[i].row = (unsigned long)rand() % rows;
    coo->data[i].col = (unsigned long)rand() % cols;
    coo->data[i].val = (float)(rand() % 10 + 1);
  }
}

void coo_reserve(COO *coo, unsigned long nnz) {
  coo->data = (COOEntry *)realloc(coo->data, nnz * sizeof(COOEntry));
  coo->nnz = nnz;
}

// Function to convert COO to CSR
//
// assumes that the csr matrix is already allocated and big enough
// it doesn't destroy the orifinal coo matrix
void coo_to_csr(COO *coo, CSR *csr) {
  csr->nnz = coo->nnz;
  csr->nrow = coo->nrow;
  csr->ncol = coo->ncol;
  // resize csr arrays
  csr->row_idx = (unsigned long *)realloc(
      csr->row_idx, (coo->nrow + 1) * sizeof(unsigned long));
  csr->col_idx =
      (unsigned long *)realloc(csr->col_idx, coo->nnz * sizeof(unsigned long));
  csr->val = (float *)realloc(csr->val, coo->nnz * sizeof(float));

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

  if (entryA->row != entryB->row) {
    return (int)entryA->row - (int)entryB->row;
  } else {
    return (int)entryA->col - (int)entryB->col;
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

  if (!mm_is_matrix(matcode) ||  !mm_is_sparse(matcode)) { //&& 
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
