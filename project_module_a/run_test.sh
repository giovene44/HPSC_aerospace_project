#!/bin/bash

# Stop execution immediately if any command fails
set -e

echo "--- Cleaning Build ---"
make clean

echo "--- Compiling Tests ---"
make test

echo "--- Running Executable ---"
# executing the binary as requested
./bin/test_momentum_x_direction_solver