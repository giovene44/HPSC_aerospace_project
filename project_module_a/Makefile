# ==========================================
# 1. CONFIGURAZIONE
# ==========================================
CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra

# Modalità build: release (default) o debug
BUILD ?= release

ifeq ($(BUILD),debug)
CXXFLAGS += -g -O0 -DDEBUG
else
CXXFLAGS += -O2
endif

# Directory
SRC_DIR  := src
TEST_DIR := tests
INC_DIR  := inc
BIN_DIR  := bin
OBJ_DIR  := obj

# File main
MAIN_FILENAME := main.cpp

# Eseguibili
MAIN_EXE := $(BIN_DIR)/main_app
TEST_EXE := $(BIN_DIR)/test_suite

# ==========================================
# 2. CONFIGURAZIONE MUPARSERX
# ==========================================
MUP_DIR       := muparserx
MUP_BUILD_DIR := $(MUP_DIR)/build
MUP_LIB       := $(MUP_BUILD_DIR)/libmuparserx.a
INCLUDES      := -I$(INC_DIR) -I$(MUP_DIR)/parser

# ==========================================
# 3. FILE SELEZIONATI AUTOMATICAMENTE
# ==========================================

ALL_SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
APP_MAIN_FILE := $(SRC_DIR)/$(MAIN_FILENAME)
CORE_SRC_FILES := $(filter-out $(APP_MAIN_FILE), $(ALL_SRC_FILES))

OBJS_CORE := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/src/%.o, $(CORE_SRC_FILES))
OBJ_MAIN  := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/src/%.o, $(APP_MAIN_FILE))
OBJS_TEST := $(patsubst $(TEST_DIR)/%.cpp, $(OBJ_DIR)/test/%.o, $(wildcard $(TEST_DIR)/*.cpp))

# ==========================================
# 4. TARGET PRINCIPALI
# ==========================================

.PHONY: all main test debug clean clean-all setup

all: main

main: setup $(MAIN_EXE)

test: setup $(TEST_EXE)

# ============================
# Modalità DEBUG corretta
# ============================
debug:
	@$(MAKE) clean
	@$(MAKE) BUILD=debug main

# ==========================================
# 5. LINKING
# ==========================================

$(MAIN_EXE): $(OBJ_MAIN) $(OBJS_CORE) $(MUP_LIB)
	@echo "[LINK APP] $@"
	$(CXX) $(OBJ_MAIN) $(OBJS_CORE) $(MUP_LIB) -o $@

$(TEST_EXE): $(OBJS_TEST) $(OBJS_CORE) $(MUP_LIB)
	@echo "[LINK TEST] $@"
	$(CXX) $(OBJS_TEST) $(OBJS_CORE) $(MUP_LIB) -o $@

# ==========================================
# 6. COMPILAZIONE
# ==========================================

$(OBJ_DIR)/src/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	@echo "[CXX SRC] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/test/%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(dir $@)
	@echo "[CXX TEST] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(MUP_LIB):
	@echo "[SUBMODULE] Building muparserx..."
	mkdir -p $(MUP_BUILD_DIR)
	cd $(MUP_BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
	cd $(MUP_BUILD_DIR) && make

setup:
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(OBJ_DIR)/src
	@mkdir -p $(OBJ_DIR)/test

# ==========================================
# 7. CLEAN
# ==========================================

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

clean-all: clean
	rm -rf $(MUP_BUILD_DIR)
