// USELESS DEFINE, MAKES HAPPY THE LINTER

#include "lib.h"
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#define ROWS (1 << 13)
#define COLS (1 << 13)
#define NNZ (1 << 24)

#define WARMUPS 40
#define REPS 500

// <file out>, <rows>, <cols>, <nnz>
int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <output_file> <rows> <cols> <nnz>\n", argv[0]);
    return -1;
  }

  COO *coo = coo_new();
  coo_generate_random(coo, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
  FILE *output_file = fopen(argv[1], "w");
  if (output_file == NULL) {
    printf("Error opening file: %s\n", argv[1]);
    return -1;
  }
  coo_write_to_file(output_file, coo);

  coo_free(coo);
  return 0;
}