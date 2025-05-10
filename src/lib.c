
/*
Entry to COO 
*/

#include <stdlib.h>
typedef struct {
    // row index (0 indexed)
    int row;
    // col index (0 indexed)
    int col;
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
    int nnz;
    // how many rows
    int nrow;
    // how many columns
    int ncol;
}COO;


/*
CSR Matrix
to obtain the number of non-
*/
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
void coo_generate_random(COO *coo, int rows, int cols, int nnz) {
    coo->nnz = nnz;
    coo->data = (COOEntry *)realloc(coo->data, nnz * sizeof(COOEntry));
    for (int i = 0; i < coo->nnz; ++i) {
        coo->data[i].row = rand() % rows;
        coo->data[i].col = rand() % cols;
        coo->data[i].val = (float)(rand() % 10 + 1);
    }
}

// Function to convert COO to CSR
// 
// assumes that the csr matrix is already allocated and big enough
// it doesn't destroy the orifinal coo matrix
void coo_to_csr(COO *coo, CSR *csr) {
    csr->nnz=coo->nnz;

    //resize csr arrays
    csr->row_idx = (int *)realloc(csr->row_idx, (coo->nrow + 1) * sizeof(int));
    csr->col_idx = (int *)realloc(csr->col_idx, coo->nnz * sizeof(int));
    csr->val = (float *)realloc(csr->val, coo->nnz * sizeof(float));

    //initialize row_idx to 0
    for (int i = 0; i <= csr->nrow; ++i)
        csr->row_idx[i] = 0;

    //count the number of non-zero elements in each row
    for (int i = 0; i < coo->nnz; ++i)
        csr->row_idx[coo->data[i].row + 1]++;

    //compute the prefix sum to get the row pointers
    for (int i = 0; i < csr->nrow; ++i)
        csr->row_idx[i + 1] += csr->row_idx[i];

    //temporary array to keep track of the current index in each row
    int *temp = (int *)malloc(csr->nrow * sizeof(int));
    
    //initialize temp to the row pointers
    for (int i = 0; i < csr->nrow; ++i)
        temp[i] = csr->row_idx[i];

    //fill the col_idx and val arrays
    for (int i = 0; i < csr->nnz; ++i) {
        int row = coo->data[i].row;
        int idx = temp[row]++;
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
    coo->data = (COOEntry *)realloc(coo->data, csr->nnz * sizeof(COOEntry));
    
    
    for (int i = 0; i < csr->nrow; ++i) {
        //for each row
        for (int j = csr->row_idx[i]; j < csr->row_idx[i + 1]; ++j) {
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
        return entryA->row - entryB->row;
    } else {
        return entryA->col - entryB->col;
    }
}

// Function to sort COO entries by row and column indices
void coo_sort_in_ascending_order(COO *coo) {
    qsort(coo->data, coo->nnz, sizeof(COOEntry), compare_cooEntry);
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