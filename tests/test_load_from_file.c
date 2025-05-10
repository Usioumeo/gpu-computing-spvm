#include"../src/headers/lib.h"
#include <stdio.h>


int main(){
    COO *coo = coo_new();
    
    //open file
    FILE* input = fopen("data/abb313/abb313.mtx", "r");
    if (input == NULL) {
        printf("Error opening file\n");
        return -1;
    }

    // load the matrix from file
    int ret;
    if ((ret=coo_from_file(input, coo))!= 0){
        printf("error loading from file %d\n", ret);
        return -1;
    }

    //verify some properties of the matrix
    if(coo->nrow != 313 || coo->ncol != 176 || coo->nnz != 1557){
        printf("Invalid headers read\n");
        return -1;
    }
    
    //free resources
    coo_free(coo);

    printf("test passed\n");
    return 0;
    
}