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
echo "scalar = sin(x)sin(t)"
echo " (I -  Dxx) (scalar) = rhs "
./bin/test_pressure_x_direction_solver


echo -e
echo "scalar = sin(y)sin(t)"
echo " (I -  Dyy) (scalar) = rhs "
./bin/test_pressure_y_direction_solver

echo -e
echo "scalar = sin(z)sin(t)"
echo " (I -  Dzz) (scalar) = rhs "
./bin/test_pressure_z_direction_solver