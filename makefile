CC=gcc

FLAGS=-O3 -Wall -Wextra -Wpedantic -mavx2 -march=native #-fsanitize=address
#-Wshadow -Wfloat-equal -Wconversion -Wsign-conversion -Wnull-dereference 
#-Wdouble-promotion -Wformat=2 -I/usr/include/suitesparse
LIBS=#-L/opt/shares/openfoam/software/OpenBLAS/0.3.23-GCC-12.3.0/lib 
INCLUDES=-I/usr/include/suitesparse  -Isrc/headers
CUDA_FLAGS=-O2  -DUSE_CUDA #-Wall -Wextra -Wpedantic -mavx2
#-Iinclude
LIB_FLAGS=-lm   -lsuitesparseconfig -lcxsparse  \
-ftree-vectorize -msse3 -mfpmath=sse -ftree-vectorizer-verbose=5 -fopt-info-vec-missed=output.miss -fopenmp #-fsanitize=address
#-fopenmp  -lopenblas


BUILD_FOLDER := build
BIN_FOLDER := $(BUILD_FOLDER)/bin
OBJ_FOLDER := $(BUILD_FOLDER)/obj
SRC_FOLDER := src
DATA_FOLDER := data
#BATCH_OUT_FOLDER := outputs

#LIB_NAME=lib

LIBS_SRC = $(wildcard $(SRC_FOLDER)/*.c)

OBJECTS = $(patsubst $(SRC_FOLDER)/%.c, $(OBJ_FOLDER)/%.o, $(LIBS_SRC)) 
#$(OBJ_FOLDER)/$(LIB_NAME).o $(OBJ_FOLDER)/mmio.o
CUDA_OBJECTS = $(patsubst $(SRC_FOLDER)/%.c, $(OBJ_FOLDER)/cuda/%.o, $(LIBS_SRC)) 

TEST_FOLDER=tests
TEST_SOURCES := $(wildcard $(TEST_FOLDER)/*.c)
TEST_CUDA_SOURCES := $(wildcard $(TEST_FOLDER)/*.cu)
TEST_OBJECTS := $(patsubst $(TEST_FOLDER)/%.c, $(OBJ_FOLDER)/tests/%.o, $(TEST_SOURCES))
TEST_BINS := $(patsubst $(TEST_FOLDER)/%.c, $(BIN_FOLDER)/tests/%, $(TEST_SOURCES))
TEST_CUDA_BINS := $(patsubst $(TEST_FOLDER)/%.cu, $(BIN_FOLDER)/cuda/%, $(TEST_CUDA_SOURCES))

all: build_tests 

clean:
	rm -rf $(BUILD_FOLDER)


############### NORMAL COMPILATION SECTION ###############

# build object files from libs
$(OBJ_FOLDER)/%.o: $(SRC_FOLDER)/%.c
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER)
	$(CC) -DUSE_OPENMP $(FLAGS) -c $< -o $@ $(LIB_FLAGS) $(INCLUDES)







############################ Download datasets ############################
# @if ! [ -f $(DATA_FOLDER)/hollywood.tar.gz ]; then \
	#	echo "Downloading dataset..."; \
	#	curl -L --output $(DATA_FOLDER)/hollywood.tar.gz https://suitesparse-collection-website.herokuapp.com/MM/LAW/hollywood-2009.tar.gz; \
	#	echo "Unpacking dataset..."; \
	#	tar -xzf $(DATA_FOLDER)/hollywood.tar.gz -C $(DATA_FOLDER); \
	#fi

datasets:
	@mkdir -p $(DATA_FOLDER)
	@if ! [ -f $(DATA_FOLDER)/abb313.tar.gz ]; then \
		echo "Downloading dataset..."; \
		curl -L --output $(DATA_FOLDER)/abb313.tar.gz https://suitesparse-collection-website.herokuapp.com/MM/HB/abb313.tar.gz; \
		echo "Unpacking dataset..."; \
		tar -xzf $(DATA_FOLDER)/abb313.tar.gz -C $(DATA_FOLDER); \
	fi
	


############################## Test BUILDING SECTION ##############################

# BUILD TEST OBJECT FILES
$(OBJ_FOLDER)/tests/%.o: $(TEST_FOLDER)/%.c
	@mkdir -p $(OBJ_FOLDER)/tests
	$(CC) -DUSE_OPENMP $(FLAGS) -c $< -o $@ $(LIBS) $(INCLUDES) $(LIB_FLAGS)

# BUILD TEST BINS
$(BIN_FOLDER)/tests/%: $(OBJ_FOLDER)/tests/%.o $(OBJECTS)
	@mkdir -p $(BIN_FOLDER)/tests
	$(CC) -DUSE_OPENMP $(FLAGS) $^ -o $@ $(LIBS) $(INCLUDES) $(LIB_FLAGS)


build_tests: $(TEST_BINS) $(TEST_CUDA_BINS)

# Add a test target
run_tests: build_tests
	@echo "Running tests..."
	@echo "Test binaries: $(TEST_BINS)"
	@for test_bin in $(TEST_BINS); do \
		echo "Running $$test_bin..."; \
		if ! ./$$test_bin; then \
			echo "Test $$test_bin failed!" >&2; \
			exit 1; \
		fi; \
	done
	@echo "All tests passed!"

############################ CUDA TESTING SECTION ##############################



$(OBJ_FOLDER)/cuda/%.o: $(SRC_FOLDER)/%.c
	@mkdir -p $(OBJ_FOLDER)/cuda
	nvcc $(CUDA_FLAGS) -c $< -o $@ $(LIBS) $(INCLUDES)

# Build test cuda object files
$(OBJ_FOLDER)/cuda/%.o: $(TEST_FOLDER)/%.cu
	@mkdir -p $(OBJ_FOLDER)/cuda
	nvcc $(CUDA_FLAGS) -c $< -o $@ $(LIBS) $(INCLUDES)

# Build cuda bins
$(BIN_FOLDER)/cuda/%: $(OBJ_FOLDER)/cuda/%.o $(CUDA_OBJECTS)
	@mkdir -p $(BIN_FOLDER)/cuda
	nvcc $(CUDA_FLAGS) $^ -o $@ $(LIBS) $(INCLUDES)

