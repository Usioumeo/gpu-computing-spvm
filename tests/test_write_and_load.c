#include"../src/headers/lib.h"
#include <stdio.h>


int main(){
    int ret;
    COO *coo = coo_new();
    coo_generate_random(coo, 100, 100, 200);
    
    //open file
    FILE* output = fopen("data/dummy.mtx", "w");
    if (output == NULL) {
        printf("Error opening file\n");
        return -1;
    }

    //write the matrix to file
    if ((ret=coo_write_to_file(output, coo))!=0){
        printf("error writing to file %d\n", ret);
        return -1;
    }
    
    //read the matrix from file
    COO *coo2 = coo_new();
    FILE* input = fopen("data/dummy.mtx", "r");
    if (input == NULL) {
        printf("Error opening file\n");
        return -1;
    }
    
    if ((ret=coo_from_file(input, coo2))!= 0){
        printf("error loading from file %d\n", ret);
        return -1;
    }

    if (coo_compare(coo, coo2)!=0){
        printf("error comparing the two matrices\n");
        return -1;
    }

    coo_free(coo);
    coo_free(coo2);
    printf("test passed\n");
    return 0;
    
}