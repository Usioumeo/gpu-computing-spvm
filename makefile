### CONFIG ZONE ###
CC=gcc
NVCC=nvcc

FLAGS=-Wall -Wextra -Wpedantic -mavx2 -march=native -DUSE_OPENMP #-fsanitize=address
#-Wshadow -Wfloat-equal -Wconversion -Wsign-conversion -Wnull-dereference 
INCLUDES=-I/usr/include/suitesparse  -Isrc/headers
CUDA_FLAGS=-DUSE_CUDA -Wno-deprecated-gpu-targets #-Wall -Wextra -Wpedantic -mavx2

LIB_FLAGS=-lm   -lsuitesparseconfig -lcxsparse  \
-ftree-vectorize -msse3 -mfpmath=sse -ftree-vectorizer-verbose=5 -fopt-info-vec-missed=output.miss -fopenmp #-fsanitize=address
#-fopenmp  -lopenblas


BUILD_FOLDER := build
TEST_SRC_FOLDER=tests
BENCH_FOLDER=benchmarks
SRC_FOLDER := src
DATA_FOLDER := data

### AUTOMAGICALLY GENERATED ZONE ###
LIBS_SRC = $(wildcard $(SRC_FOLDER)/*.c)
#TEST_OBJECTS := $(patsubst $(TEST_FOLDER)/%.c, $(OBJ_FOLDER)/tests/%.o, $(TEST_SOURCES))

all:

clean:
	rm -rf $(BUILD_FOLDER)
# 1=path to parent folder, 2 compiler, 3 flags, 4 OPT
define BUILD_OBJ_LIB_TEMPLATE
$(1)/obj_lib/%$(4).o: $(SRC_FOLDER)/%.c
	@mkdir -p $(1)/obj_lib
	$(2) $(if $(4),-$(4),) $(3) -c $$< -o $$@ $(INCLUDES)
endef

# 1=path to parent folder, 2 compiler, 3 flags, 4 path source folders, 5 extension, 6 OPT
define BUILD_OBJ_TEMPLATE
$(1)/obj/%$(6).o: $(4)/%.$(5)
	@echo $$@ 
	@mkdir -p $(1)/obj 
	$(2) $(if $(6),-$(6),) $(3) -c $$< -o $$@ $(INCLUDES)
endef

# 1=path to parent folder, 2 compiler, 3 flags, 4 obj dependencies, 5 linker flags , 6 OPT
define BUILD_BIN_TEMPLATE
$(1)/bins/%$(6): $(1)/obj/%$(6).o $(4)
	@echo $$@ 
	@mkdir -p $(1)/bins 
	$(2) $(if $(6),-$(6),) $(3)  $$^ -o $$@ $(INCLUDES) $(5)
endef


### INSTANCING PART####


#std_tests
$(eval $(call BUILD_OBJ_LIB_TEMPLATE,$(BUILD_FOLDER)/tests/std,$(CC),$(FLAGS)))
$(eval $(call BUILD_OBJ_TEMPLATE,$(BUILD_FOLDER)/tests/std,$(CC),$(FLAGS),$(TEST_SRC_FOLDER),c))

OBJ_LIB_DEPS := $(patsubst $(SRC_FOLDER)/%.c, $(BUILD_FOLDER)/tests/std/obj_lib/%.o, $(LIBS_SRC))

$(eval $(call BUILD_BIN_TEMPLATE,$(BUILD_FOLDER)/tests/std,$(CC),$(FLAGS),$(OBJ_LIB_DEPS),$(LIB_FLAGS)))

TEST_SOURCES := $(wildcard $(TEST_SRC_FOLDER)/*.c)
STD_TESTS := $(patsubst $(TEST_SRC_FOLDER)/%.c, $(BUILD_FOLDER)/tests/std/bins/%, $(TEST_SOURCES))
build_std_tests: $(STD_TESTS)


#cuda_tests
$(eval $(call BUILD_OBJ_LIB_TEMPLATE,$(BUILD_FOLDER)/tests/cuda,$(NVCC),$(CUDA_FLAGS)))
$(eval $(call BUILD_OBJ_TEMPLATE,$(BUILD_FOLDER)/tests/cuda,$(NVCC),$(CUDA_FLAGS),$(TEST_SRC_FOLDER),cu))
OBJ_LIB_DEPS_CUDA := $(patsubst $(SRC_FOLDER)/%.cu, $(BUILD_FOLDER)/tests/cuda/obj_lib/%.o, $(LIBS_SRC))
$(eval $(call BUILD_BIN_TEMPLATE,$(BUILD_FOLDER)/tests/cuda,$(NVCC),$(CUDA_FLAGS),$(OBJ_LIB_DEPS_CUDA)))

TEST_SOURCES_CUDA := $(wildcard $(TEST_SRC_FOLDER)/*.cu)
CUDA_TESTS := $(patsubst $(TEST_SRC_FOLDER)/%.cu, $(BUILD_FOLDER)/tests/cuda/bins/%, $(TEST_SOURCES_CUDA))
build_cuda_tests: $(CUDA_TESTS)

build_tests: build_std_tests build_cuda_tests

run_tests: build_tests
	@echo "Running tests..."
	@echo "Test binaries: $(STD_TESTS)"
	@for test_bin in $(STD_TESTS); do \
		echo "Running $$test_bin..."; \
		if ! ./$$test_bin; then \
			echo "Test $$test_bin failed!" >&2; \
			exit 1; \
		fi; \
	done
	@echo "Test binaries: $(CUDA_TESTS)"
	@for test_bin in $(CUDA_TESTS); do \
		echo "Running $$test_bin..."; \
		if ! ./$$test_bin; then \
			echo "Test $$test_bin failed!" >&2; \
			exit 1; \
		fi; \
	done
	@echo "All tests passed!"


#std_bench
$(eval $(call BUILD_OBJ_LIB_TEMPLATE,$(BUILD_FOLDER)/bench/std,$(CC),$(FLAGS),O0))
$(eval $(call BUILD_OBJ_TEMPLATE,$(BUILD_FOLDER)/bench/std,$(CC),$(FLAGS),$(BENCH_FOLDER),c,O0))
OBJ_LIB_DEPSO0 := $(patsubst $(SRC_FOLDER)/%.c, $(BUILD_FOLDER)/bench/std/obj_lib/%O0.o, $(LIBS_SRC))

$(eval $(call BUILD_BIN_TEMPLATE,$(BUILD_FOLDER)/bench/std,$(CC),$(FLAGS),$(OBJ_LIB_DEPSO0),$(LIB_FLAGS),O0))
#$(eval $(call BUILD_BIN_TEMPLATE,$(BUILD_FOLDER)/tests/std,$(CC),$(FLAGS),$(OBJ_LIB_DEPS),$(LIB_FLAGS)))



BENCH_SOURCES := $(wildcard $(BENCH_FOLDER)/*.c)
STD_BENCH := $(patsubst $(BENCH_FOLDER)/%.c, $(BUILD_FOLDER)/bench/std/bins/%O0, $(BENCH_SOURCES))
debug:
	@echo $(STD_BENCH)
build_std_bench: $(STD_BENCH)
