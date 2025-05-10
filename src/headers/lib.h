#ifndef LIBCSR_H
#define LIBCSR_H

#include <stdio.h>
#include <stdlib.h>
typedef struct {
    // row index (0 indexed)
    unsigned long row;
    // col index (0 indexed)
    unsigned long col;
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
    unsigned long nnz;
    // how many rows
    unsigned long nrow;
    // how many columns
    unsigned long ncol;
}COO;


/*
CSR Matrix
to obtain the number of non-
*/
typedef struct{
    // one row index for each row +1 (and value refers to col_idx and val)
    unsigned long *row_idx;
    // array of columns, there is one for each non-zero element
    unsigned long *col_idx;
    // array of values, there is one for each non-zero element
    float *val;
    // how many rows
    unsigned long nrow;
    // how many columns
    unsigned long ncol;
    // how many non-zero elements (its not strictly necessary to store it, it can be computed as row_idx[nrow])
    unsigned long nnz;
}CSR;

// Function to generate random sparse matrix in COO format
void coo_generate_random(COO *coo, unsigned long rows, unsigned long cols, unsigned long nnz);

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

void coo_reserve(COO *coo, unsigned long nnz);

int coo_from_file(FILE *input, COO *coo);


// compare function for qsort
int compare_cooEntry(const void *a, const void *b);

int coo_write_to_file(FILE *output, COO *coo);

int coo_compare(COO *coo1, COO *coo2);
#endif