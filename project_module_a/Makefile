# ==========================================
# 1. CONFIGURAZIONE
# ==========================================

# MPI support: set USE_MPI=1 to enable parallel execution
USE_MPI ?= 0
NP ?= 4

ifeq ($(USE_MPI),1)
CXX      := mpicxx
CXXFLAGS := -std=c++17 -Wall -Wextra -DUSE_MPI
RUN_CMD  := mpirun -np $(NP)
else
CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra
RUN_CMD  :=
endif

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

# File main dell'applicazione principale
MAIN_FILENAME := main.cpp

# Eseguibile principale
MAIN_EXE := $(BIN_DIR)/main_app

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

# --- SORGENTI PRINCIPALI ---
ALL_SRC_FILES  := $(wildcard $(SRC_DIR)/*.cpp)
APP_MAIN_FILE  := $(SRC_DIR)/$(MAIN_FILENAME)
# Core files sono tutti tranne il main.cpp (per evitare conflitti di main multipli nei test)
CORE_SRC_FILES := $(filter-out $(APP_MAIN_FILE), $(ALL_SRC_FILES))

OBJS_CORE := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/src/%.o, $(CORE_SRC_FILES))
OBJ_MAIN  := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/src/%.o, $(APP_MAIN_FILE))

# --- SORGENTI TEST ---
# Troviamo tutti i file .cpp in tests/
TEST_SRCS := $(wildcard $(TEST_DIR)/*.cpp)

# Definiamo gli oggetti dei test
OBJS_TEST := $(patsubst $(TEST_DIR)/%.cpp, $(OBJ_DIR)/test/%.o, $(TEST_SRCS))

# Definiamo gli ESEGUIBILI dei test.
# Esempio: tests/test_math.cpp -> bin/test_math
TEST_EXES := $(patsubst $(TEST_DIR)/%.cpp, $(BIN_DIR)/%, $(TEST_SRCS))

# ==========================================
# 4. TARGET PRINCIPALI
# ==========================================

.PHONY: all main test debug clean clean-all setup run-tests run info

all: main

main: setup $(MAIN_EXE)

info:
	@echo "Build configuration:"
	@echo "  CXX         = $(CXX)"
	@echo "  USE_MPI     = $(USE_MPI)"
	@echo "  BUILD       = $(BUILD)"
	@echo "  NP          = $(NP)"
	@echo "  RUN_CMD     = $(RUN_CMD)"
	@echo ""
	@echo "To build with MPI: make USE_MPI=1"
	@echo "To run with MPI:   make USE_MPI=1 NP=4 run"

# Il target 'test' ora compila TUTTI gli eseguibili separati
test: setup $(TEST_EXES)
	@echo "Build dei test completata. Eseguibili in $(BIN_DIR)/"

# Utility per eseguire tutti i test generati in sequenza
run-tests: test
	@for test_bin in $(TEST_EXES); do \
		echo "Running $$test_bin..."; \
		$(RUN_CMD) $$test_bin || exit 1; \
		echo "--------------------------------"; \
	done

# Run main application
run: main
	@echo "Running main application..."
	$(RUN_CMD) $(MAIN_EXE)

# ============================
# Modalità DEBUG
# ============================
debug:
	@$(MAKE) clean
	@$(MAKE) BUILD=debug main

# ==========================================
# 5. LINKING
# ==========================================

# Link dell'applicazione principale
$(MAIN_EXE): $(OBJ_MAIN) $(OBJS_CORE) $(MUP_LIB)
	@echo "[LINK APP] $@"
	$(CXX) $(OBJ_MAIN) $(OBJS_CORE) $(MUP_LIB) -o $@

# Link dinamico per i TEST separati
# La regola % matcha il nome del file (senza estensione) dentro bin/
# Dipende dal corrispondente .o in obj/test/, dai file CORE e dalla LIB
$(BIN_DIR)/%: $(OBJ_DIR)/test/%.o $(OBJS_CORE) $(MUP_LIB)
	@echo "[LINK TEST] $@"
	$(CXX) $< $(OBJS_CORE) $(MUP_LIB) -o $@

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