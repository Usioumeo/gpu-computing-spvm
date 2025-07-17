
/*
Entry to COO
*/

#include "mmio.h"
#include "uthash.h"
#include <immintrin.h>
#include <lib.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct {
  UT_hash_handle hh;
  uint64_t key;
} CooDedup;

uint64_t key_gen(unsigned row, unsigned col) {
  uint64_t key = 0;
  key |= (uint64_t)row << 32;
  key |= (uint64_t)col;
  return key;
}
// Function to generate random sparse matrix in COO format
void coo_generate_random(COO *coo, unsigned rows, unsigned cols, unsigned nnz) {
  if ((unsigned long)nnz > (unsigned long)rows * cols) {
    printf("Error: nnz cannot be greater than rows * cols\n");
    exit(1);
  }
  CooDedup *set = NULL, *entry;
  CooDedup *tmp_set = (CooDedup *)malloc(nnz * sizeof(CooDedup));
  coo->nrow = rows;
  coo->ncol = cols;
  coo_reserve(coo, nnz);

  for (unsigned i = 0; i < coo->nnz; ++i) {
    coo->data[i].row = (unsigned)rand() % rows;
    coo->data[i].col = (unsigned)rand() % cols;
    uint64_t key = key_gen(coo->data[i].row, coo->data[i].col);
    CooDedup *entry;
    HASH_FIND(hh, set, &key, sizeof(uint64_t), entry);
    if (entry) {
      // Duplicate found, decrement nnz and continue
      i--;
      continue;
    }
    tmp_set[i].key = key;
    entry = &tmp_set[i];
    HASH_ADD(hh, set, key, sizeof(uint64_t), entry);
    coo->data[i].val = (float)(rand() % 2001 - 1000) / 1000.0f; // Random value
  }
  HASH_CLEAR(hh, set);
  free(tmp_set);
}

// Function to reserve memory for COO matrix
void coo_reserve(COO *coo, unsigned nnz) {
  coo->data = (COOEntry *)realloc(coo->data, nnz * sizeof(COOEntry));
  coo->nnz = nnz;
}

//#ifndef USE_CUDA
// Function to reserve memory for CSR matrix
void csr_reserve(CSR *csr, unsigned nnz, unsigned nrow) {
  csr->nnz = nnz;
  csr->nrow = nrow;
  // resize csr arrays
  csr->row_idx =
      (unsigned *)realloc(csr->row_idx, (nrow + 1) * sizeof(unsigned));
  csr->col_idx = (unsigned *)realloc(csr->col_idx, nnz * sizeof(unsigned));
  csr->val = (float *)realloc(csr->val, nnz * sizeof(float));
}

// Function to free the memory allocated for CSR matrix
// IT ALSO FREES THE POINTER
void csr_free(CSR *csr) {
  free(csr->row_idx);
  free(csr->col_idx);
  free(csr->val);
  csr->row_idx = NULL;
  csr->col_idx = NULL;
  csr->val = NULL;
  free(csr);
}

// Function to create a new empty CSR matrix
CSR *csr_new() {
  CSR *csr = (CSR *)malloc(sizeof(CSR));
  csr->row_idx = NULL;
  csr->col_idx = NULL;
  csr->val = NULL;
  csr->nnz = 0;
  csr->nrow = 0;
  csr->ncol = 0;
  return csr;
}

//#endif

// Function to convert COO to CSR
//
// assumes that the csr matrix is already allocated and big enough
// it doesn't destroy the orifinal coo matrix
void coo_to_csr(COO *coo, CSR *csr) {
  csr->ncol = coo->ncol;
  // resize csr arrays
  csr_reserve(csr, coo->nnz, coo->nrow);

  // initialize row_idx to 0
  for (unsigned i = 0; i <= csr->nrow; ++i)
    csr->row_idx[i] = 0;

  // count the number of non-zero elements in each row
  for (unsigned i = 0; i < coo->nnz; ++i)
    csr->row_idx[coo->data[i].row + 1]++;

  // compute the prefix sum to get the row pointers
  for (unsigned i = 0; i < csr->nrow; ++i)
    csr->row_idx[i + 1] += csr->row_idx[i];

  // temporary array to keep track of the current index in each row
  unsigned *temp = (unsigned *)malloc(csr->nrow * sizeof(unsigned));

  // initialize temp to the row pointers
  for (unsigned i = 0; i < csr->nrow; ++i)
    temp[i] = csr->row_idx[i];

  // fill the col_idx and val arrays
  for (unsigned i = 0; i < csr->nnz; ++i) {
    unsigned row = coo->data[i].row;
    unsigned idx = temp[row]++;
    csr->col_idx[idx] = coo->data[i].col;
    csr->val[idx] = coo->data[i].val;
  }
  // free the temporary array
  free(temp);
  // sort the CSR matrix in ascending order
  csr_sort_in_ascending_order(*csr);
}

// Function to convert CSR to COO

// it doesn't destroy the orifinal csr matrix
void csr_to_coo(CSR *csr, COO *coo) {
  // init coo
  coo->nrow = csr->nrow;
  coo->ncol = csr->ncol;
  coo_reserve(coo, csr->nnz);

  for (unsigned i = 0; i < csr->nrow; ++i) {
    // for each row
    for (unsigned j = csr->row_idx[i]; j < csr->row_idx[i + 1]; ++j) {
      // for each entry in the row
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

  if (entryA->col != entryB->col) {
    return (int)entryA->col - (int)entryB->col;
  } else {
    return (int)entryA->row - (int)entryB->row;
  }
}

// Function to sort COO entries by row and column indices
void coo_sort_in_ascending_order(COO *coo) {
  qsort(coo->data, (long unsigned)coo->nnz, sizeof(COOEntry), compare_cooEntry);
}

// Function to free the memory allocated for COO matrix
// IT ALSO FREES THE POINTER
void coo_free(COO *coo) {
  free(coo->data);
  // coo->data = NULL;
  free(coo);
}

// Function to create a new empty COO matrix
COO *coo_new() {
  COO *coo = (COO *)malloc(sizeof(COO));
  coo->data = NULL;
  coo->nnz = 0;
  coo->nrow = 0;
  coo->ncol = 0;
  return coo;
}


void read_char_blob(char **pos, char *out){
  char *cur = *pos;
  //skip spaces
  while(*cur<=' '){
    cur++;
  }
  //read until next space
  while(*cur>' '){
    *out++ = *cur++;
  }
  *out = '\0';
  *pos = cur;
}
unsigned read_unsigned(char **pos) {
  char out_buffer[128];
  read_char_blob(pos, out_buffer);
  return (unsigned)atoi(out_buffer);
}
float read_float(char **pos) {
  char out_buffer[128];
  read_char_blob(pos, out_buffer);
  return (float)atof(out_buffer);
}

int coo_from_file(FILE *input, COO *coo) {
  MM_typecode matcode;

  if (mm_read_banner(input, &matcode) != 0) {
    printf("Could not process Matrix Market banner.\n");
    exit(1);
  }

  /*  This is how one can screen matrix types if their application */
  /*  only supports a subset of the Matrix Market data types.      */

  if (!mm_is_matrix(matcode) || !mm_is_sparse(matcode)) { //&&
    printf("Sorry, this application does not support ");
    printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    exit(1);
  }

  /* find out size of sparse matrix .... */
  int M, N, nnz;
  if (mm_read_mtx_crd_size(input, &M, &N, &nnz) != 0)
    exit(1);
  coo->nrow = M;
  coo->ncol = N;
  coo->nnz = nnz;

  long where_to_read  = ftell(input);
  fseek(input, 0, SEEK_END);
  long fsize = ftell(input);
  fseek(input, where_to_read, SEEK_SET);
  char * tmp_buffer = (char *)malloc(fsize - where_to_read);
  //try to read
  if (fread(tmp_buffer, fsize - where_to_read, 1, input) != 1) {
    printf("Error reading file\n");
    free(tmp_buffer);
    return -1;
  }
  fclose(input);


  /* reseve memory for matrices */
  coo_reserve(coo, coo->nnz);

  /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
  /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
  /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
  char * buffer_pointer = tmp_buffer;
  printf("reading %u rows, %u cols, %u nnz\n", coo->nrow, coo->ncol,
         coo->nnz);
  fflush(stdout);
  for (unsigned int i = 0; i < coo->nnz; i++) {
    /*int adv;
    int c=sscanf(buffer_pointer, "%u %u %f\n%n", &coo->data[i].row, &coo->data[i].col,
               &coo->data[i].val, &adv);*/
    coo->data[i].row = read_unsigned(&buffer_pointer);
    coo->data[i].col = read_unsigned(&buffer_pointer);
    coo->data[i].val = read_float(&buffer_pointer);
    /*printf("read %u %u %f\n", coo->data[i].row, coo->data[i].col,
           coo->data[i].val);*/
    //buffer_pointer += adv;

    coo->data[i].row--; /* adjust from 1-based to 0-based */
    coo->data[i].col--;
  }
  free(tmp_buffer);
  printf("scanfato\n");
  fflush(stdout);
  return 0;
}

int coo_write_to_file(FILE *output, COO *coo) {
  MM_typecode matcode;
  unsigned i;

  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_coordinate(&matcode);
  mm_set_real(&matcode);

  mm_write_banner(output, matcode);
  mm_write_mtx_crd_size(output, coo->nrow, coo->ncol, coo->nnz);

  /* NOTE: matrix market files use 1-based indices, i.e. first element
    of a vector has index 1, not 0.  */

  for (i = 0; i < coo->nnz; i++)
    fprintf(output, "%u %u %10.3f\n", coo->data[i].row + 1,
            coo->data[i].col + 1, coo->data[i].val);
  fclose(output);
  return 0;
}

int coo_compare(COO *coo1, COO *coo2) {
  if (coo1->nnz != coo2->nnz || coo1->nrow != coo2->nrow ||
      coo1->ncol != coo2->ncol) {
    printf("coo1 %u %u %u\n", coo1->nrow, coo1->ncol, coo1->nnz);
    printf("coo2 %u %u %u\n", coo2->nrow, coo2->ncol, coo2->nnz);
    return 1;
  }
  coo_sort_in_ascending_order(coo1);
  coo_sort_in_ascending_order(coo2);

  for (unsigned i = 0; i < coo1->nnz; ++i) {
    if (coo1->data[i].row != coo2->data[i].row ||
        coo1->data[i].col != coo2->data[i].col ||
        coo1->data[i].val - coo2->data[i].val > 0.00000001) {
      printf("coo1 %u %u %f\n", coo1->data[i].row, coo1->data[i].col,
             coo1->data[i].val);
      printf("coo2 %u %u %f\n", coo2->data[i].row, coo2->data[i].col,
             coo2->data[i].val);
      return 1;
    }
  }
  return 0;
}

int spmv_coo(COO coo, unsigned n, float *input_vec, float *output_vec) {
  if (n != coo.ncol) {
    return 1;
  }
  for (unsigned i = 0; i < coo.nrow; i++) {
    output_vec[i] = 0.0;
  }
  for (unsigned i = 0; i < coo.nnz; ++i) {
    output_vec[coo.data[i].row] += coo.data[i].val * input_vec[coo.data[i].col];
  }
  return 0;
}

// default implementation, it should be the correct version
int spmv_csr(CSR csr, unsigned n, float *restrict input_vec, float *restrict output_vec) {
  if (n != csr.ncol) {
    return 1;
  }
  for (unsigned i = 0; i < csr.nrow; ++i) {
    output_vec[i] = 0.0;
    float out = 0.0;
    for (unsigned j = csr.row_idx[i]; j < csr.row_idx[i + 1]; ++j) {
      out+= csr.val[j] * input_vec[csr.col_idx[j]];
    }
    output_vec[i] = out;
  }

  return 0;
}
#define BLOCK_SIZE 2
int spmv_csr_block(CSR csr, unsigned n, float *input_vec, float *output_vec) {
  if (n != csr.ncol) {
    return 1;
  }
  for (unsigned block_start = 0; block_start < csr.nrow;
       block_start += BLOCK_SIZE) {
    unsigned block_max = (block_start + BLOCK_SIZE < csr.nrow)
                             ? (block_start + BLOCK_SIZE)
                             : csr.nrow;
    for (unsigned i = block_start; i < block_max; ++i) {
      output_vec[i] = 0.0;

      for (unsigned j = csr.row_idx[i]; j < csr.row_idx[i + 1]; ++j) {
        output_vec[i] += csr.val[j] * input_vec[csr.col_idx[j]];
      }
    }
  }

  return 0;
}

typedef struct {
  float val;
  unsigned col;
} TmpStruct;

int compare_tmpStruct(const void *a, const void *b) {
  TmpStruct *entryA = (TmpStruct *)a;
  TmpStruct *entryB = (TmpStruct *)b;

  return (int)entryA->col - (int)entryB->col;
}

void csr_sort_in_ascending_order(CSR csr) {
  TmpStruct *tmp = (TmpStruct *)malloc(csr.nrow * sizeof(TmpStruct));
  for (unsigned i = 0; i < csr.nrow; ++i) {
    unsigned int start = csr.row_idx[i];
    unsigned int end = csr.row_idx[i + 1];
    for (unsigned j = start; j < end; ++j) {
      tmp[j - start].val = csr.val[j];
      tmp[j - start].col = csr.col_idx[j];
    }
    qsort(tmp, end - start, sizeof(TmpStruct), compare_tmpStruct);
    for (unsigned j = start; j < end; ++j) {
      csr.val[j] = tmp[j - start].val;
      csr.col_idx[j] = tmp[j - start].col;
    }
  }
  free(tmp);
}
#ifdef USE_OPENMP
#include <immintrin.h>
int spmv_csr_openmp(CSR csr, unsigned n, float *input_vec, float *output_vec) {
  if (n != csr.ncol) {
    return 1;
  }
#pragma omp parallel for schedule(static, 1)
  for (unsigned i = 0; i < csr.nrow; ++i) {
    output_vec[i] = 0.0;
// no race conditions, because each thread writes on a different index of the
// output vector
#pragma omp simd
    for (unsigned j = csr.row_idx[i]; j < csr.row_idx[i + 1]; ++j) {
      output_vec[i] += csr.val[j] * input_vec[csr.col_idx[j]];
    }
  }
  return 0;
}

int spmv_csr_openmp_simd(CSR csr, unsigned n, float *input_vec,
                         float *output_vec) {
  if (n != csr.ncol) {
    return 1;
  }
#pragma omp parallel for schedule(static, 1)
  for (unsigned i = 0; i < csr.nrow; ++i) {
    output_vec[i] = 0.0;
    unsigned start = csr.row_idx[i];
    unsigned start2 = (start + 7) & ~7;
    unsigned end = csr.row_idx[i + 1];
    unsigned end2 = end & ~7;

    __m256 cumulate = _mm256_setzero_ps();

    for (unsigned j = start; j < end2; j += 8) {
      // load 8 offsets
      __m256 vec1 = _mm256_loadu_ps(&csr.val[j]);
      __m256i indices = _mm256_loadu_si256((const __m256i_u *)&csr.col_idx[j]);
      __m256 vec2 = _mm256_i32gather_ps(input_vec, indices, 4);
      __m256 product = _mm256_mul_ps(vec1, vec2);
      cumulate = _mm256_add_ps(cumulate, product);
    }

    for (int j = 0; j < 8; j++) {
      output_vec[i] += cumulate[j];
    }
    for (unsigned k = start; k < start2; k++) {
      output_vec[i] += csr.val[k] * input_vec[csr.col_idx[k]];
    }
    for (unsigned j = end2; j < end; ++j) {
      output_vec[i] += csr.val[j] * input_vec[csr.col_idx[j]];
    }
  }
  return 0;
}

int spmv_csr_order(CSR csr, unsigned n, float *input_vec, float *output_vec) {
  if (n != csr.ncol) {
    return 1;
  }
#pragma omp parallel for
  for (unsigned i = 0; i < csr.nrow; ++i) {
    output_vec[i] = 0;
// no race conditions, because each thread writes on a different index of the
// output vector
#pragma omp simd
    for (unsigned j = csr.row_idx[i]; j < csr.row_idx[i + 1]; ++j) {
      output_vec[i] += csr.val[j] * input_vec[csr.col_idx[j]];
    }
  }
  return 0;
}

#endif

int relative_error_compare(float *a, float *b, unsigned n) {
  for (unsigned j = 0; j < n; j++) {

    if (!((a[j] - b[j])/(fabs(a[j])+0.001) < 0.001 ||fabs(a[j] - b[j])<0.001)) {
      printf("The two output are not the same: %f!=%f (%u)\n", a[j],
             b[j], j);
      return -1;
    }
  }
  return 0;
}

int write_bin_to_file(CSR *csr, const char *filename) {
  FILE *output = fopen(filename, "wb");
  if (output == NULL) {
    fprintf(stderr, "Error opening file for writing: %s\n", filename);
    return -1;
  }
  fwrite(&csr->nrow, sizeof(unsigned), 1, output);
  fwrite(&csr->ncol, sizeof(unsigned), 1, output);
  fwrite(&csr->nnz, sizeof(unsigned), 1, output);
  fwrite(csr->row_idx, sizeof(unsigned), csr->nrow + 1, output);
  fwrite(csr->col_idx, sizeof(unsigned), csr->nnz, output);
  fwrite(csr->val, sizeof(float), csr->nnz, output);
  fclose(output);
  return 0;
}
int read_bin_to_csr(const char *filename, CSR *csr) {
  FILE *input = fopen(filename, "rb");
  if (input == NULL) {
    fprintf(stderr, "Error opening file for reading: %s\n", filename);
    return -1;
  }
  fread(&csr->nrow, sizeof(unsigned), 1, input);
  fread(&csr->ncol, sizeof(unsigned), 1, input);
  fread(&csr->nnz, sizeof(unsigned), 1, input);
  csr->row_idx = (unsigned *)malloc((csr->nrow + 1) * sizeof(unsigned));
  csr->col_idx = (unsigned *)malloc(csr->nnz * sizeof(unsigned));
  csr->val = (float *)malloc(csr->nnz * sizeof(float));
  fread(csr->row_idx, sizeof(unsigned), csr->nrow + 1,  input);
  fread(csr->col_idx, sizeof(unsigned), csr->nnz, input);
  fread(csr->val, sizeof(float), csr->nnz, input);
  fclose(input);
  return 0;
}
