#!/bin/bash

# Stop execution immediately if any command fails
set -e

echo "--- Cleaning Build ---"
make clean
rm -rf ./Output

echo "--- Compiling Tests ---"
make -j OMP=1


echo "--- Running Executable ---"
# executing the binary as requested
echo -e
OMP_NUM_THREADS=8 OMP_PROC_BIND=close ./bin/main_app