#!/bin/bash

# Stop execution immediately if any command fails
set -e

echo "--- Cleaning Build ---"
make clean

echo "--- Compiling Tests ---"
make -j OMP=1


echo "--- Running Executable ---"
# executing the binary as requested
echo -e
echo "vector_1 = sin(x)sin(dt),    vector_2 = sin(x)sin(dt+dt) \n"
echo " (I - Gamma * Dxx) (delta_vector) = delta_rhs "
OMP_NUM_THREADS=8 OMP_PROC_BIND=close ./bin/main_app