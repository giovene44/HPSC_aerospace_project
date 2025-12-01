#!/bin/bash

# Stop execution immediately if any command fails
set -e

echo "--- Cleaning Build ---"
make clean

echo "--- Compiling Tests ---"
make test

echo "--- Running Executable ---"
# executing the binary as requested
echo -e
./bin/test_brinkman_system

