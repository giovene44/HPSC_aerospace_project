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
.PHONY: all main test debug profile clean clean-all clean-profile setup run-tests analyze-profile

all: main

main: setup $(MAIN_EXE)

# Il target 'test' ora compila TUTTI gli eseguibili separati
test: setup $(TEST_EXES)
	@echo "Build dei test completata. Eseguibili in $(BIN_DIR)/"

# Utility per eseguire tutti i test generati in sequenza
run-tests: test
	@for test_bin in $(TEST_EXES); do \
		echo "Running $$test_bin..."; \
		./$$test_bin || exit 1; \
		echo "--------------------------------"; \
	done

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
$(MAIN_EXE): $(OBJ_MAIN) $(OBJS_CORE) $(MUP_LIB)
	@echo "[LINK APP] $@"
	$(CXX) $(LDFLAGS) $(OBJ_MAIN) $(OBJS_CORE) $(MUP_LIB) -o $@

# Link dinamico per i TEST separati
# La regola % matcha il nome del file (senza estensione) dentro bin/
# Dipende dal corrispondente .o in obj/test/, dai file CORE e dalla LIB
$(BIN_DIR)/%: $(OBJ_DIR)/test/%.o $(OBJS_CORE) $(MUP_LIB)
	@echo "[LINK TEST] $@"
	$(CXX) $(LDFLAGS) $< $(OBJS_CORE) $(MUP_LIB) -o $@

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

clean-profile:
	rm -f gmon.out analysis.txt
	@echo "File di profiling rimossi"

clean-all: clean clean-profile
	rm -rf $(MUP_BUILD_DIR)