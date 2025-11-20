# ==========================================
# 1. CONFIGURAZIONE
# ==========================================
CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2

# Directory
SRC_DIR  := src
TEST_DIR := tests
INC_DIR  := inc
BIN_DIR  := bin
OBJ_DIR  := obj

# --- IMPORTANTE: IL NOME DEL TUO FILE CON IL MAIN ---
# Inserisci qui il nome esatto del file in src/ che contiene int main()
MAIN_FILENAME := main.cpp

# Nomi eseguibili
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
# 3. LOGICA DI SELEZIONE FILE (IL FILTRO)
# ==========================================

# 1. Trova TUTTI i file .cpp in src/
ALL_SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)

# 2. Definisce il percorso completo del file main
APP_MAIN_FILE := $(SRC_DIR)/$(MAIN_FILENAME)

# 3. Crea una lista dei file "Core" (tutto src tranne il main)
#    Questi servono SIA all'app CHE ai test
CORE_SRC_FILES := $(filter-out $(APP_MAIN_FILE), $(ALL_SRC_FILES))

# 4. Genera liste oggetti
# Oggetti della logica comune (Header implementati)
OBJS_CORE := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/src/%.o, $(CORE_SRC_FILES))
# Oggetto specifico del main (solo per l'app)
OBJ_MAIN  := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/src/%.o, $(APP_MAIN_FILE))
# Oggetti dei test
OBJS_TEST := $(patsubst $(TEST_DIR)/%.cpp, $(OBJ_DIR)/test/%.o, $(wildcard $(TEST_DIR)/*.cpp))

# ==========================================
# 4. TARGET PRINCIPALI
# ==========================================

.PHONY: all main test clean clean-all setup

all: main

# L'APP finale unisce: Oggetto Main + Oggetti Core + Muparserx
main: setup $(MAIN_EXE)

# I TEST uniscono: Oggetti Test + Oggetti Core + Muparserx (ESCLUSO il main dell'app)
test: setup $(TEST_EXE)

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

# Compila file src/
$(OBJ_DIR)/src/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	@echo "[CXX SRC] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compila file test/
$(OBJ_DIR)/test/%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(dir $@)
	@echo "[CXX TEST] $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compila Muparserx (Submodule)
$(MUP_LIB):
	@echo "[SUBMODULE] Building muparserx..."
	mkdir -p $(MUP_BUILD_DIR)
	cd $(MUP_BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
	cd $(MUP_BUILD_DIR) && make

setup:
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(OBJ_DIR)/src
	@mkdir -p $(OBJ_DIR)/tests

# ==========================================
# 7. PULIZIA
# ==========================================

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

clean-all: clean
	rm -rf $(MUP_BUILD_DIR)