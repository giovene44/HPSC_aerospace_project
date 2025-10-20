# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O2
DEBUG_FLAGS = -std=c++11 -Wall -Wextra -g -O0

# Directories
INCLUDE_DIR = inc
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin
TEST_DIR = tests

# Detect all source and test files
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
TEST_SRCS = $(wildcard $(TEST_DIR)/test_*.cpp)
EXCLUDE_TEST_SRCS = $(SRC_DIR)/main.cpp $(SRC_DIR)/navies_stokes_brinkman.cpp # TODO: Remove this once the main.cpp and navies_stokes_brinkman.cpp are removed from the tests
TEST_LINK_SRCS = $(filter-out $(EXCLUDE_TEST_SRCS), $(SRC_FILES))

# Generate corresponding binary names
TEST_BINS = $(patsubst $(TEST_DIR)/%.cpp, $(BIN_DIR)/%, $(TEST_SRCS))
TEST_DEBUG_BINS = $(patsubst $(TEST_DIR)/%.cpp, $(BIN_DIR)/%_debug, $(TEST_SRCS))

# Optional variable to run one test
# Example: make run TEST=test_matrix3d
TEST ?=

# Default target
all: directories $(TEST_BINS)

directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

# Build release test binaries (linking source files)
$(BIN_DIR)/%: $(TEST_DIR)/%.cpp $(TEST_LINK_SRCS) | directories
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) $(TEST_LINK_SRCS) $< -o $@
	@echo "Build complete: $@"

# Build debug test binaries (linking source files)
$(BIN_DIR)/%_debug: $(TEST_DIR)/%.cpp $(TEST_LINK_SRCS) | directories
	$(CXX) $(DEBUG_FLAGS) -I$(INCLUDE_DIR) $(TEST_LINK_SRCS) $< -o $@
	@echo "Debug build complete: $@"

debug: directories $(TEST_DEBUG_BINS)

# Run tests
run: all
ifeq ($(TEST),)
	@echo "Running all tests..."
	@for t in $(TEST_BINS); do echo "==> $$t"; ./$$t; echo ""; done
else
	@echo "Running test: $(TEST)"
	@./$(BIN_DIR)/$(TEST)
endif

# Run debug tests
run-debug: debug
ifeq ($(TEST),)
	@echo "Running all debug tests..."
	@for t in $(TEST_DEBUG_BINS); do echo "==> $$t"; ./$$t; echo ""; done
else
	@echo "Running debug test: $(TEST)_debug"
	@./$(BIN_DIR)/$(TEST)_debug
endif

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "Cleaned build artifacts"

rebuild: clean all

help:
	@echo "Available targets:"
	@echo "  all          - Build all test files (default)"
	@echo "  debug        - Build all test files in debug mode"
	@echo "  run          - Run all tests or one test (make run TEST=<name>)"
	@echo "  run-debug    - Run all debug tests or one debug test"
	@echo "  clean        - Remove build artifacts"
	@echo "  rebuild      - Clean and rebuild"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Example:"
	@echo "  make run TEST=test_matrix3d"

.PHONY: all directories debug run run-debug clean rebuild help
