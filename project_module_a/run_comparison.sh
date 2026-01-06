#!/bin/bash

echo "================================================================"
echo "Serial vs Parallel Schur Complement Solver Comparison"
echo "================================================================"
echo ""

echo "1. Running in serial mode (1 process)..."
echo "================================================================"
mpirun -np 1 ./bin/test_schur_solver
echo ""

echo "2. Running with 2 MPI processes..."
echo "================================================================"
mpirun -np 2 ./bin/test_schur_solver
echo ""

echo "3. Running with 4 MPI processes..."
echo "================================================================"
mpirun -np 4 ./bin/test_schur_solver
echo ""

echo "4. Running with 8 MPI processes..."
echo "================================================================"
mpirun -np 8 ./bin/test_schur_solver

