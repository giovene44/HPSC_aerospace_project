# ==========================================
# 1. CONFIGURAZIONE
# ==========================================
#   make                      seriale
#   make OMP=1                OpenMP
#   make MPI=1 OMP=1          MPI + OpenMP (eseguire con mpirun -n P ./bin/main_app)
#   make BUILD=debug          debug (-O0 -g)
#   make ARCH=                disabilita -march=native (es. nodi di calcolo eterogenei)
CXX      ?= g++
ifeq ($(origin CXX),default)
CXX      := g++
endif
CXXFLAGS := -std=c++17 -Wall -Wextra
LDFLAGS  :=
UNAME_S  := $(shell uname -s)

# Enable MPI by passing MPI=1 to make (default: off)
# When MPI is enabled, use mpicxx as compiler and define USE_MPI
MPI ?= 0
ifeq ($(MPI),1)
CXX      := mpicxx
CXXFLAGS += -DUSE_MPI
endif

# Enable OpenMP by passing OMP=1 to make (default: off)
OMP ?= 0
ifeq ($(OMP),1)
ifeq ($(UNAME_S),Darwin)
# Apple clang: OpenMP through Homebrew's libomp (brew install libomp)
LIBOMP   ?= $(shell brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)
CXXFLAGS += -Xpreprocessor -fopenmp -I$(LIBOMP)/include
LDFLAGS  += -L$(LIBOMP)/lib -lomp
else
CXXFLAGS += -fopenmp
LDFLAGS  += -fopenmp
endif
endif

# Modalità build: release (default) o debug
BUILD ?= release
ARCH  ?= -march=native
ifeq ($(BUILD),debug)
CXXFLAGS += -g -O0 -DDEBUG
else
CXXFLAGS += -O3 $(ARCH)
endif

# Profiling con gprof: impostare PROFILE=1 per abilitarlo
PROFILE ?= 0
ifeq ($(PROFILE),1)
CXXFLAGS += -pg
LDFLAGS  += -pg
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

# Cambiando configurazione (MPI, OMP, BUILD, ...) tutto viene ricompilato: i flag correnti
# sono salvati in obj/.build_flags e, se cambiano, oggetti ed eseguibili vengono rimossi
# (non ci si affida ai timestamp: GNU make 3.81 li confronta al secondo).
FLAGS_STAMP := $(OBJ_DIR)/.build_flags
BUILD_ID    := $(CXX) $(CXXFLAGS) $(LDFLAGS)
$(shell mkdir -p $(OBJ_DIR); echo '$(BUILD_ID)' | cmp -s - $(FLAGS_STAMP) || \
  { rm -rf $(OBJ_DIR)/src $(OBJ_DIR)/test $(BIN_DIR); echo '$(BUILD_ID)' > $(FLAGS_STAMP); })

# ==========================================
# 2. INCLUDE
# ==========================================
# Le espressioni analitiche dell'input sono compilate da inc/Expression.hpp:
# muparserx non e' piu' necessario.
INCLUDES      := -I$(INC_DIR)

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
.PHONY: all main test check debug profile clean clean-all clean-profile setup run-tests analyze-profile

all: main

main: setup $(MAIN_EXE)

# Il target 'test' ora compila TUTTI gli eseguibili separati
test: setup $(TEST_EXES)
	@echo "Build dei test completata. Eseguibili in $(BIN_DIR)/"

# Verifica automatica del solutore (solutore tridiagonale + convergenza MMS).
# Con MPI=1 viene eseguita su CHECK_NP processi (default 4).
CHECK_NP ?= 4
check: setup $(BIN_DIR)/test_solver_mms
ifeq ($(MPI),1)
	mpirun --oversubscribe -n $(CHECK_NP) ./$(BIN_DIR)/test_solver_mms
else
	./$(BIN_DIR)/test_solver_mms
endif

# Utility per eseguire tutti i test generati in sequenza
run-tests: test
	@for test_bin in $(TEST_EXES); do \
		echo "Running $$test_bin..."; \
		./$$test_bin || exit 1; \
		echo "--------------------------------"; \
	done

# Comparison executables
compare: bin/run_serial_solver bin/run_parallel_solver
	@echo "Built comparison tools: bin/run_serial_solver and bin/run_parallel_solver"

bin/run_serial_solver: tests/run_serial_solver.cpp
	@mkdir -p bin
	@echo "[CXX] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

bin/run_parallel_solver: tests/run_parallel_solver.cpp $(OBJS_CORE)
	@mkdir -p bin
	@echo "[CXX] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(OBJS_CORE) $(LDFLAGS) -o $@

# ============================
# Modalità DEBUG
# ============================
debug:
	@$(MAKE) clean
	@$(MAKE) BUILD=debug main

# ============================
# Modalità PROFILING
# ============================
profile:
	@$(MAKE) clean
	@$(MAKE) PROFILE=1 main
	@echo ""
	@echo "=========================================="
	@echo "Profiling build completata!"
	@echo "=========================================="
	@echo "Passi successivi:"
	@echo "1. Esegui il programma:      ./$(MAIN_EXE) [args]"
	@echo "2. Verrà generato:           gmon.out"
	@echo "3. Analizza i risultati:     gprof $(MAIN_EXE) gmon.out"
	@echo "4. Per output leggibile:     gprof $(MAIN_EXE) gmon.out > analysis.txt"
	@echo "=========================================="

# Shortcut per analizzare il profiling dopo l'esecuzione
analyze-profile:
	@if [ ! -f gmon.out ]; then \
		echo "Errore: gmon.out non trovato!"; \
		echo "Devi prima eseguire il programma compilato con PROFILE=1"; \
		exit 1; \
	fi
	@echo "Generazione analisi profiling..."
	gprof $(MAIN_EXE) gmon.out > analysis.txt
	@echo "Analisi salvata in analysis.txt"
	@echo "Mostrando le prime 50 righe:"
	@echo "=========================================="
	@head -n 50 analysis.txt

# ==========================================
# 5. LINKING
# ==========================================
# Link dell'applicazione principale
$(MAIN_EXE): $(OBJ_MAIN) $(OBJS_CORE)
	@echo "[LINK APP] $@"
	$(CXX) $(OBJ_MAIN) $(OBJS_CORE) $(LDFLAGS) -o $@

# Link dinamico per i TEST separati
# La regola % matcha il nome del file (senza estensione) dentro bin/
# Dipende dal corrispondente .o in obj/test/, dai file CORE e dalla LIB
$(BIN_DIR)/%: $(OBJ_DIR)/test/%.o $(OBJS_CORE)
	@echo "[LINK TEST] $@"
	$(CXX) $< $(OBJS_CORE) $(LDFLAGS) -o $@

# ==========================================
# 6. COMPILAZIONE
# ==========================================
$(OBJ_DIR)/src/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	@echo "[CXX SRC] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

$(OBJ_DIR)/test/%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(dir $@)
	@echo "[CXX TEST] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

setup:
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(OBJ_DIR)/src
	@mkdir -p $(OBJ_DIR)/test

# ==========================================
# 7. CLEAN
# ==========================================
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

clean-profile:
	rm -f gmon.out analysis.txt
	@echo "File di profiling rimossi"

clean-all: clean clean-profile
# dipendenze dagli header (generate con -MMD)
-include $(wildcard $(OBJ_DIR)/src/*.d $(OBJ_DIR)/test/*.d)

# keep the test objects (otherwise removed as intermediate files)
.PRECIOUS: $(OBJ_DIR)/test/%.o
