#ifndef LIBCSR_H
#define LIBCSR_H

#include <stdlib.h>
typedef struct {
    // row index (0 indexed)
    int row;
    // col index (0 indexed)
    int col;
    // value contained
    float val; 
} COOEntry;

typedef struct {
    // multiple COO entries
    COOEntry *data;
    //non-zero elements, aka number of COO entries,
    int nnz;
    // how many rows
    int nrow;
    // how many columns
    int ncol;
}COO;


typedef struct{
    // one row index for each row +1 (and value refers to col_idx and val)
    int *row_idx;
    // array of columns, there is one for each non-zero element
    int *col_idx;
    // array of values, there is one for each non-zero element
    float *val;
    // how many rows
    int nrow;
    // how many columns
    int ncol;
    // how many non-zero elements (its not strictly necessary to store it, it can be computed as row_idx[nrow])
    int nnz;
}CSR;

// Function to generate random sparse matrix in COO format
void coo_generate_random(COO *coo, int rows, int cols, int nnz);

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

#endif