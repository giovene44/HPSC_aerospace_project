#!/bin/bash

# Stop execution immediately if any command fails
set -e

echo "--- Cleaning Build ---"
make clean

echo "--- Compiling Tests (OpenMP) ---"
make test OMP=1

echo "--- Running Executables (OpenMP) ---"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
echo "Using OMP_NUM_THREADS=$OMP_NUM_THREADS OMP_PROC_BIND=$OMP_PROC_BIND"

echo
echo "scalar = sin(x)sin(t)"
echo " (I -  Dxx) (scalar) = rhs "
OMP_NUM_THREADS=$OMP_NUM_THREADS OMP_PROC_BIND=$OMP_PROC_BIND ./bin/test_pressure_x_direction_solver

echo
echo "scalar = sin(y)sin(t)"
echo " (I -  Dyy) (scalar) = rhs "
OMP_NUM_THREADS=$OMP_NUM_THREADS OMP_PROC_BIND=$OMP_PROC_BIND ./bin/test_pressure_y_direction_solver

echo
echo "scalar = sin(z)sin(t)"
echo " (I -  Dzz) (scalar) = rhs "
OMP_NUM_THREADS=$OMP_NUM_THREADS OMP_PROC_BIND=$OMP_PROC_BIND ./bin/test_pressure_z_direction_solver