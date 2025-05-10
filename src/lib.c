
/*
Entry to COO 
*/

#include <stdio.h>
#include <stdlib.h>
#include "headers/lib.h"


// Function to generate random sparse matrix in COO format
void coo_generate_random(COO *coo, unsigned long rows, unsigned long cols, unsigned long nnz) {
    coo->nnz = nnz;
    coo->nrow = rows;
    coo->ncol = cols;
    coo->data = (COOEntry *)realloc(coo->data, nnz * sizeof(COOEntry));
    for (unsigned long i = 0; i < coo->nnz; ++i) {
        coo->data[i].row = (unsigned long)rand() % rows;
        coo->data[i].col = (unsigned long)rand() % cols;
        coo->data[i].val = (float)(rand() % 10 + 1);
    }
}

// Function to convert COO to CSR
// 
// assumes that the csr matrix is already allocated and big enough
// it doesn't destroy the orifinal coo matrix
void coo_to_csr(COO *coo, CSR *csr) {
    csr->nnz=coo->nnz;
    csr->nrow=coo->nrow;
    csr->ncol=coo->ncol;
    //resize csr arrays
    csr->row_idx = (unsigned long *)realloc(csr->row_idx, (coo->nrow + 1) * sizeof(unsigned long));
    csr->col_idx = (unsigned long *)realloc(csr->col_idx, coo->nnz * sizeof(unsigned long));
    csr->val = (float *)realloc(csr->val, coo->nnz * sizeof(float));

    //initialize row_idx to 0
    for (unsigned long i = 0; i <= csr->nrow; ++i)
        csr->row_idx[i] = 0;

    //count the number of non-zero elements in each row
    for (unsigned long i = 0; i < coo->nnz; ++i)
        csr->row_idx[coo->data[i].row + 1]++;

    //compute the prefix sum to get the row pointers
    for (unsigned long i = 0; i < csr->nrow; ++i)
        csr->row_idx[i + 1] += csr->row_idx[i];

    //temporary array to keep track of the current index in each row
    unsigned long *temp = (unsigned long*)malloc(csr->nrow * sizeof(unsigned long));
    
    //initialize temp to the row pointers
    for (unsigned long i = 0; i < csr->nrow; ++i)
        temp[i] = csr->row_idx[i];

    //fill the col_idx and val arrays
    for (unsigned long i = 0; i < csr->nnz; ++i) {
        unsigned long row = coo->data[i].row;
        unsigned long idx = temp[row]++;
        csr->col_idx[idx] = coo->data[i].col;
        csr->val[idx] = coo->data[i].val;
    }
    //free the temporary array
    free(temp);
}

// Function to convert CSR to COO

// it doesn't destroy the orifinal csr matrix
void csr_to_coo(CSR *csr, COO *coo) {
    //init coo 
    coo->nnz = csr->nnz;
    coo->data = (COOEntry *)realloc(coo->data, (long unsigned) csr->nnz * sizeof(COOEntry));
    
    
    for (unsigned long i = 0; i < csr->nrow; ++i) {
        //for each row
        for (unsigned long j = csr->row_idx[i]; j < csr->row_idx[i + 1]; ++j) {
            //for each entry in the row
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
    qsort(coo->data, (long unsigned) coo->nnz, sizeof(COOEntry), compare_cooEntry);
}

// Function to free the memory allocated for COO matrix
// IT ALSO FREES THE POINTER
void coo_free(COO *coo) {
    free(coo->data);
    //coo->data = NULL;
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
COO * coo_new(){
    COO *coo = (COO *)malloc(sizeof(COO));
    coo->data = NULL;
    coo->nnz = 0;
    coo->nrow = 0;
    coo->ncol = 0;
    return coo;
}

// Function to create a new empty CSR matrix
CSR* csr_new(){
    CSR *csr = (CSR *)malloc(sizeof(CSR));
    csr->row_idx = NULL;
    csr->col_idx = NULL;
    csr->val = NULL;
    csr->nnz = 0;
    csr->nrow = 0;
    csr->ncol = 0;
    return csr; 
}