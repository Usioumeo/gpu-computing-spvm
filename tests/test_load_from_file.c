#include"../src/headers/lib.h"
#include <stdio.h>


int main(){
    COO *coo = coo_new();
    
    FILE* input = fopen("data/abb313/abb313.mtx", "r");
    if (input == NULL) {
        printf("Error opening file\n");
        return -1;
    }
    int ret;
    if ((ret=coo_from_file(input, coo))!= 0){
        printf("error loading from file %d\n", ret);
        return -1;
    }
    if(coo->nrow != 313 || coo->ncol != 176 || coo->nnz != 1557){
        printf("Invalid headers read\n");
        return -1;
    }
    
    coo_free(coo);

    printf("test passed\n");
    return 0;
    
}