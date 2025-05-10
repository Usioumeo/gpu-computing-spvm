/*
CSR project

Send a report (up 4 pages, ideally 2) with:
• Your name – student ID – email address
• Link to git repository

Structure:
• Section one – problem description, description of the algorithm and pseudo-code.
• Section two – Experimental setup:
▪ Hardware used
▪ Plots describing the performance and the variation of the performance. (e.g., x-axis matrix dimensions, y-axis effective-bandwidth)
• Refenences





1) setup ssh


- benchmarking with random, or with a matrix from a file
- test adds instrumentation, and runs multiple tests in order to verify correctness
- use INSTRUMENTAL/DEBUG in order to add to debug (debug information and other)

from csr to COO
COO to CSR
from csr to dummy matrix
load_csr
load_coo

General Matrix multiplication (dummy matrix)
cpu dummy product
csr product cpu (dummy) + optimization
csr product gpu (dummy) + optimization

# si possono usare le atomics
*/


#include"headers/lib.h"
int main(){

    COO *coo = (COO *)malloc(sizeof(COO));
    coo_generate_random(coo, 10, 10, 10);

    
}