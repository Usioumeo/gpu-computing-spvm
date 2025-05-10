CC=gcc

FLAGS=-g -fsanitize=address -O0 -Wall -Wextra -Wpedantic -Wshadow -Wfloat-equal -Wconversion -Wsign-conversion -Wnull-dereference -Wdouble-promotion -Wformat=2
LIBS=-L/opt/shares/openfoam/software/OpenBLAS/0.3.23-GCC-12.3.0/lib 
INCLUDES=-I/opt/shares/openfoam/software/OpenBLAS/0.3.23-GCC-12.3.0/include -Iinclude
LIB_FLAGS=-lm -lopenblas 


BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src
#BATCH_OUT_FOLDER := outputs

MAIN_NAME=main
MAIN_BIN=$(MAIN_NAME)
MAIN_SRC=$(SRC_FOLDER)/$(MAIN_NAME).c


LIB_NAME=lib



OBJECTS = $(OBJ_FOLDER)/$(LIB_NAME).o

all: $(BIN_FOLDER)/$(MAIN_BIN) $(BIN_FOLDER)/$(MAIN_TEST) 


# build object files

$(OBJ_FOLDER)/$(LIB_NAME).o: $(SRC_FOLDER)/$(LIB_NAME).c
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER)
	$(CC) $(FLAGS) -c $(SRC_FOLDER)/$(LIB_NAME).c -o $@ $(LIB_FLAGS)


# main exec
$(BIN_FOLDER)/$(MAIN_BIN): $(MAIN_SRC) $(OBJECTS)
	@mkdir -p $(BIN_FOLDER)
	$(CC) $(FLAGS) $^ -o $@ $(LIBS) $(INCLUDES) $(LIB_FLAGS)



clean:
	rm -rf $(BIN_FOLDER) $(OBJ_FOLDER)




datasets:
	@curl -L --output "hollywood" https://suitesparse-collection-website.herokuapp.com/MM/LAW/hollywood-2009.tar.gz


TEST_FOLDER := tests
TEST_SOURCES := $(wildcard $(TEST_FOLDER)/*.c)
TEST_OBJECTS := $(patsubst $(TEST_FOLDER)/%.c, $(OBJ_FOLDER)/%.o, $(TEST_SOURCES))
TEST_BINS := $(patsubst $(TEST_FOLDER)/%.c, $(BIN_FOLDER)/%, $(TEST_SOURCES))

# Add a test target
test: $(TEST_BINS)
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

# Build test object files
$(OBJ_FOLDER)/%.o: $(TEST_FOLDER)/%.c
	@mkdir -p $(OBJ_FOLDER)
	$(CC) $(FLAGS) -c $< -o $@ $(LIBS) $(INCLUDES) $(LIB_FLAGS)

# Build test executables
$(BIN_FOLDER)/%: $(OBJ_FOLDER)/%.o $(OBJECTS)
	@mkdir -p $(BIN_FOLDER)
	$(CC) $(FLAGS) $^ -o $@ $(LIBS) $(INCLUDES) $(LIB_FLAGS)
