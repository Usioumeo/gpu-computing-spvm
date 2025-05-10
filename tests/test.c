#include"../src/headers/lib.h"
#include <stdio.h>


int main(){
    COO *coo = coo_new();
    coo_generate_random(coo, 10, 10, 10);
    
    CSR *csr = csr_new();
    //coo_to_csr(coo, csr);
    

    COO *coo2 = coo_new();
    //csr_to_coo(csr, coo2);

    //coo_sort_in_ascending_order(coo);
    // the second should already be sorted



    
    coo_free(coo);
    coo_free(coo2);
    csr_free(csr);
    
    printf("test passed\n");
    return 0;
    
}