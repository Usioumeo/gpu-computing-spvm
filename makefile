CC=gcc

FLAGS=-O2 -Wall -Wextra -Wpedantic -mavx2 #-fsanitize=address
#-Wshadow -Wfloat-equal -Wconversion -Wsign-conversion -Wnull-dereference 
#-Wdouble-promotion -Wformat=2 -I/usr/include/suitesparse
LIBS=#-L/opt/shares/openfoam/software/OpenBLAS/0.3.23-GCC-12.3.0/lib 
INCLUDES=-I/usr/include/suitesparse -Iinclude
LIB_FLAGS=-lm   -lsuitesparseconfig -lcxsparse  \
-ftree-vectorize -msse3 -mfpmath=sse -ftree-vectorizer-verbose=5 -fopt-info-vec-missed=output.miss -fopenmp #-fsanitize=address
#-fopenmp  -lopenblas

BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src
DATA_FOLDER := data
#BATCH_OUT_FOLDER := outputs

MAIN_NAME=main
MAIN_BIN=$(MAIN_NAME)
MAIN_SRC=$(SRC_FOLDER)/$(MAIN_NAME).c


LIB_NAME=lib

OBJECTS = $(OBJ_FOLDER)/$(LIB_NAME).o $(OBJ_FOLDER)/mmio.o


TEST_FOLDER=tests
TEST_SOURCES := $(wildcard $(TEST_FOLDER)/*.c)
TEST_OBJECTS := $(patsubst $(TEST_FOLDER)/%.c, $(OBJ_FOLDER)/tests/%.o, $(TEST_SOURCES))
TEST_BINS := $(patsubst $(TEST_FOLDER)/%.c, $(BIN_FOLDER)/tests/%, $(TEST_SOURCES))


all: $(BIN_FOLDER)/$(MAIN_BIN) $(BIN_FOLDER)/$(MAIN_TEST) 




# build object files

$(OBJ_FOLDER)/%.o: $(SRC_FOLDER)/%.c
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER)
	$(CC) $(FLAGS) -c $< -o $@ $(LIB_FLAGS)


# main exec
$(BIN_FOLDER)/$(MAIN_BIN): $(MAIN_SRC) $(OBJECTS)
	@mkdir -p $(BIN_FOLDER)
	$(CC) $(FLAGS) $^ -o $@ $(LIBS) $(INCLUDES) $(LIB_FLAGS)



clean:
	rm -rf $(BIN_FOLDER) $(OBJ_FOLDER) $(DATA_FOLDER) output.miss



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
	



# Build test object files
$(OBJ_FOLDER)/tests/%.o: $(TEST_FOLDER)/%.c
	@mkdir -p $(OBJ_FOLDER)/tests
	$(CC) $(FLAGS) -c $< -o $@ $(LIBS) $(INCLUDES) $(LIB_FLAGS)

# Build test bins
$(BIN_FOLDER)/tests/%: $(OBJ_FOLDER)/tests/%.o $(OBJECTS)
	@mkdir -p $(BIN_FOLDER)/tests
	$(CC) $(FLAGS) $^ -o $@ $(LIBS) $(INCLUDES) $(LIB_FLAGS)


build_tests: $(TEST_BINS) datasets

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

