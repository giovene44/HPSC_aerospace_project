# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
DEBUG_FLAGS = -std=c++17 -Wall -Wextra -g -O0

# Directories
INCLUDE_DIR = inc
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin
TEST_DIR = tests

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

# All source files (including main)
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)

# Exclude main.cpp from being linked into test binaries
SRC_NO_MAIN = $(filter-out $(SRC_DIR)/main.cpp, $(SRC_FILES))

# Detect test source files
TEST_SRCS = $(wildcard $(TEST_DIR)/test_*.cpp)

# Binaries
MAIN_BIN = $(BIN_DIR)/main
TEST_BINS = $(patsubst $(TEST_DIR)/%.cpp, $(BIN_DIR)/%, $(TEST_SRCS))
TEST_DEBUG_BINS = $(patsubst $(TEST_DIR)/%.cpp, $(BIN_DIR)/%_debug, $(TEST_SRCS))

# Optional variable to run one test
TEST ?=

# ---------------------------------------------------------------------------
# Default target
# ---------------------------------------------------------------------------
all: directories $(MAIN_BIN) $(TEST_BINS)

directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

# ---------------------------------------------------------------------------
# Main solver build
# ---------------------------------------------------------------------------
$(MAIN_BIN): $(SRC_FILES) | directories
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) $(SRC_FILES) -o $@
	@echo "✅ Build complete: $@"

# ---------------------------------------------------------------------------
# Test builds (release and debug)
# ---------------------------------------------------------------------------
$(BIN_DIR)/%: $(TEST_DIR)/%.cpp $(SRC_NO_MAIN) | directories
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) $(SRC_NO_MAIN) $< -o $@
	@echo "✅ Test build complete: $@"

$(BIN_DIR)/%_debug: $(TEST_DIR)/%.cpp $(SRC_NO_MAIN) | directories
	$(CXX) $(DEBUG_FLAGS) -I$(INCLUDE_DIR) $(SRC_NO_MAIN) $< -o $@
	@echo "🐞 Debug test build complete: $@"

debug: directories $(TEST_DEBUG_BINS)

# ---------------------------------------------------------------------------
# Run targets
# ---------------------------------------------------------------------------
run: $(MAIN_BIN)
	@echo "Running main program..."
	@./$(MAIN_BIN)

run-tests: all
ifeq ($(TEST),)
	@echo "Running all tests..."
	@for t in $(TEST_BINS); do echo "==> $$t"; ./$$t; echo ""; done
else
	@echo "Running test: $(TEST)"
	@./$(BIN_DIR)/$(TEST)
endif

run-debug: debug
ifeq ($(TEST),)
	@echo "Running all debug tests..."
	@for t in $(TEST_DEBUG_BINS); do echo "==> $$t"; ./$$t; echo ""; done
else
	@echo "Running debug test: $(TEST)_debug"
	@./$(BIN_DIR)/$(TEST)_debug
endif

# ---------------------------------------------------------------------------
# Utility targets
# ---------------------------------------------------------------------------
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "🧹 Cleaned build artifacts"

rebuild: clean all

help:
	@echo "Available targets:"
	@echo "  all          - Build main program and all tests"
	@echo "  run          - Run main program"
	@echo "  run-tests    - Run all test binaries or one test (make run-tests TEST=test_name)"
	@echo "  debug        - Build all tests in debug mode"
	@echo "  run-debug    - Run debug tests"
	@echo "  clean        - Remove build artifacts"
	@echo "  rebuild      - Clean and rebuild everything"
	@echo ""
	@echo "Example:"
	@echo "  make run-tests TEST=test_matrix3d"

.PHONY: all directories debug run run-tests run-debug clean rebuild help
