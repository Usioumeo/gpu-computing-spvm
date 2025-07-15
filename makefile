### CONFIG ZONE ###
CC=gcc
NVCC=nvcc
ADDITIONAL_FLAGS :=# -fsanitize=address -g
FLAGS=-Wall -Wextra -Wpedantic -mavx2 -march=native -DUSE_OPENMP -fopenmp $(ADDITIONAL_FLAGS)#-fsanitize=address
#-Wshadow -Wfloat-equal -Wconversion -Wsign-conversion -Wnull-dereference 
INCLUDES=-I/usr/include/suitesparse  -Isrc/headers
CUDA_FLAGS=-DUSE_CUDA -Wno-deprecated-gpu-targets -rdc=true -lcusparse #-arch=sm_86 #-Wall -Wextra -Wpedantic -mavx2

LIB_FLAGS=-lm -ftree-vectorize -msse3 -mfpmath=sse -ftree-vectorizer-verbose=5 -fopt-info-vec-missed=output.miss  #-fsanitize=address
#-fopenmp  -lopenblas -lsuitesparseconfig -lcxsparse  


BUILD_FOLDER := build
TEST_SRC_FOLDER=tests
BENCH_FOLDER=benchmarks
SRC_FOLDER := src
DATA_FOLDER := data

### AUTOMAGICALLY GENERATED ZONE ###
MAKEFLAGS += -j
LIBS_SRC = $(wildcard $(SRC_FOLDER)/*.c)
LIBS_SRC_CUDA = $(wildcard $(SRC_FOLDER)/*.cu)
#TEST_OBJECTS := $(patsubst $(TEST_FOLDER)/%.c, $(OBJ_FOLDER)/tests/%.o, $(TEST_SOURCES))

all: build_tests build_bench datasets
	@echo "All tests and benchmarks built successfully!"
#TODO add data
clean:
	rm -rf $(BUILD_FOLDER)
# 1=path to parent folder, 2 compiler, 3 flags, 4 extension, 5 OPT
define BUILD_OBJ_LIB_TEMPLATE
$(1)/obj_lib/%$(5).o: $(SRC_FOLDER)/%.$(4)
	@mkdir -p $(1)/obj_lib
	$(2) $(if $(5),-$(5),) $(3) -c $$< -o $$@ $(INCLUDES)
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
$(eval $(call BUILD_OBJ_LIB_TEMPLATE,$(BUILD_FOLDER)/tests/std,$(CC),$(FLAGS),c))
$(eval $(call BUILD_OBJ_TEMPLATE,$(BUILD_FOLDER)/tests/std,$(CC),$(FLAGS),$(TEST_SRC_FOLDER),c))

OBJ_LIB_DEPS := $(patsubst $(SRC_FOLDER)/%.c, $(BUILD_FOLDER)/tests/std/obj_lib/%.o, $(LIBS_SRC))

$(eval $(call BUILD_BIN_TEMPLATE,$(BUILD_FOLDER)/tests/std,$(CC),$(FLAGS),$(OBJ_LIB_DEPS),$(LIB_FLAGS)))

TEST_SOURCES := $(wildcard $(TEST_SRC_FOLDER)/*.c)
STD_TESTS := $(patsubst $(TEST_SRC_FOLDER)/%.c, $(BUILD_FOLDER)/tests/std/bins/%, $(TEST_SOURCES))
build_std_tests: $(STD_TESTS)


#cuda_tests
$(eval $(call BUILD_OBJ_LIB_TEMPLATE,$(BUILD_FOLDER)/tests/cuda,$(NVCC),$(CUDA_FLAGS),c))
$(eval $(call BUILD_OBJ_LIB_TEMPLATE,$(BUILD_FOLDER)/tests/cuda,$(NVCC),$(CUDA_FLAGS),cu))
$(eval $(call BUILD_OBJ_TEMPLATE,$(BUILD_FOLDER)/tests/cuda,$(NVCC),$(CUDA_FLAGS),$(TEST_SRC_FOLDER),cu))
OBJ_LIB_DEPS_CUDA := $(patsubst $(SRC_FOLDER)/%.cu, $(BUILD_FOLDER)/tests/cuda/obj_lib/%.o, $(LIBS_SRC_CUDA)) $(patsubst $(SRC_FOLDER)/%.c, $(BUILD_FOLDER)/tests/cuda/obj_lib/%.o, $(LIBS_SRC))
$(eval $(call BUILD_BIN_TEMPLATE,$(BUILD_FOLDER)/tests/cuda,$(NVCC),$(CUDA_FLAGS),$(OBJ_LIB_DEPS_CUDA)))

TEST_SOURCES_CUDA := $(wildcard $(TEST_SRC_FOLDER)/*.cu)
CUDA_TESTS := $(patsubst $(TEST_SRC_FOLDER)/%.cu, $(BUILD_FOLDER)/tests/cuda/bins/%,$(TEST_SOURCES_CUDA))
build_cuda_tests: $(CUDA_TESTS)

build_tests: build_std_tests build_cuda_tests

run_tests: build_tests datasets
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
define BUILD_BENCH_TEMPLATE
$(eval $(call BUILD_OBJ_LIB_TEMPLATE,$(BUILD_FOLDER)/bench/std,$(CC),$(FLAGS),c,O$(1)))
$(eval $(call BUILD_OBJ_TEMPLATE,$(BUILD_FOLDER)/bench/std,$(CC),$(FLAGS),$(BENCH_FOLDER),c,O$(1)))
$(eval OBJ_LIB_DEPSO$(1) := $(patsubst $(SRC_FOLDER)/%.c, $(BUILD_FOLDER)/bench/std/obj_lib/%O$(1).o, $(LIBS_SRC)))
$(eval $(call BUILD_BIN_TEMPLATE,$(BUILD_FOLDER)/bench/std,$(CC),$(FLAGS),$(OBJ_LIB_DEPSO$(1)),$(LIB_FLAGS),O$(1)))


$(eval $(call BUILD_OBJ_LIB_TEMPLATE,$(BUILD_FOLDER)/bench/cuda,$(NVCC),$(CUDA_FLAGS),c,O$(1)))
$(eval $(call BUILD_OBJ_LIB_TEMPLATE,$(BUILD_FOLDER)/bench/cuda,$(NVCC),$(CUDA_FLAGS),cu,O$(1)))
$(eval $(call BUILD_OBJ_TEMPLATE,$(BUILD_FOLDER)/bench/cuda,$(NVCC),$(CUDA_FLAGS),$(BENCH_FOLDER),cu,O$(1)))

$(eval OBJ_LIB_DEPS_CUDAO$(1) := $(patsubst $(SRC_FOLDER)/%.cu, $(BUILD_FOLDER)/bench/cuda/obj_lib/%O$(1).o, $(LIBS_SRC_CUDA)) $(patsubst $(SRC_FOLDER)/%.c, $(BUILD_FOLDER)/bench/cuda/obj_lib/%O$(1).o, $(LIBS_SRC)))
$(eval $(call BUILD_BIN_TEMPLATE,$(BUILD_FOLDER)/bench/cuda,$(NVCC),$(CUDA_FLAGS),$(OBJ_LIB_DEPS_CUDAO$(1)),,O$(1)))
endef

$(foreach opt,0 1 2 3,$(eval $(call BUILD_BENCH_TEMPLATE,$(opt))))

BENCH_SOURCES := $(wildcard $(BENCH_FOLDER)/*.c)
BENCH_CUDA_SOURCES := $(wildcard $(BENCH_FOLDER)/*.cu)

STD_BENCH := \
	$(foreach opt,0 1 2 3, \
		$(patsubst $(BENCH_FOLDER)/%.c, $(BUILD_FOLDER)/bench/std/bins/%O$(opt), $(BENCH_SOURCES)) \
	)

CUDA_BENCH := \
	$(foreach opt,0 1 2 3, \
		$(patsubst $(BENCH_FOLDER)/%.cu, $(BUILD_FOLDER)/bench/cuda/bins/%O$(opt), $(BENCH_CUDA_SOURCES)) \
	)



build_std_bench: $(STD_BENCH)
build_cuda_bench: $(CUDA_BENCH)

build_bench: build_std_bench build_cuda_bench


define DATASET_URLS
mawi_201512020330=https://suitesparse-collection-website.herokuapp.com/MM/MAWI/mawi_201512020330.tar.gz
mawi_201512020000=https://suitesparse-collection-website.herokuapp.com/MM/MAWI/mawi_201512020000.tar.gz
nlpkkt240=https://suitesparse-collection-website.herokuapp.com/MM/Schenk/nlpkkt240.tar.gz
bump_2911=https://suitesparse-collection-website.herokuapp.com/MM/Janna/Bump_2911.tar.gz

endef
export DATASET_URLS
DATASETS := $(foreach l,$(DATASET_URLS),$(firstword $(subst =, ,$(l))))

# Helper to get URL by dataset name
get-url = $(word 2, $(subst =, ,$(filter $(1)=%, $(DATASET_URLS))))

# Rule to download and extract a dataset
$(DATA_FOLDER)/%.tar.gz:
	@mkdir -p $(DATA_FOLDER)
	@url="$(call get-url,$*)"; \
	if [ -z "$$url" ]; then \
		echo "No URL found for $*"; exit 1; \
	fi; \
	echo "Downloading $$url..."; \
	curl -L --output $@ "$$url"
	@echo "Unpacking $@..."; \
	tar -xzf $@ -C $(DATA_FOLDER)

$(DATA_FOLDER)/random2.mtx: $(BUILD_FOLDER)/bench/std/bins/0_generate_dataO3
	$(BUILD_FOLDER)/bench/std/bins/0_generate_dataO3 $(DATA_FOLDER)/random2.mtx 35991342 35991342 37242710


datasets: $(addprefix $(DATA_FOLDER)/, $(addsuffix .tar.gz, $(DATASETS))) $(DATA_FOLDER)/random2.mtx

