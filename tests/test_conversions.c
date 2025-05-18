#include"../src/headers/lib.h"
#include <stdio.h>


int main(){
    COO *coo = coo_new();
    coo_generate_random(coo, 10, 10, 100);
    

    coo_sort_in_ascending_order(coo);

    CSR *csr = csr_new();
    coo_to_csr(coo, csr);

    COO *coo2 = coo_new();
    csr_to_coo(csr, coo2);
    // the second should already be sorted

    
    /*for(unsigned i = 0; i < coo->nnz; ++i){
        printf("coo %u %u %f\n", coo->data[i].row, coo->data[i].col, coo->data[i].val);
    }
    printf("\n");
    for(unsigned i = 0; i < coo2->nnz; ++i){
        printf("coo2 %u %u %f\n", coo2->data[i].row, coo2->data[i].col, coo2->data[i].val);
    }*/

    if (coo_compare(coo, coo2)!=0){
        printf("error comparing the two matrices\n");
        return -1;
    }

    
    coo_free(coo);
    coo_free(coo2);
    csr_free(csr);
    
    printf("test passed\n");
    return 0;
    
}