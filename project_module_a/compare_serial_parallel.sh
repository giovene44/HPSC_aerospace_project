#!/bin/bash
# Script to compare serial Thomas solver vs parallel Schur complement solver

echo "================================================================"
echo "Serial vs Parallel Solver Comparison"
echo "================================================================"
echo ""
echo "This script compares:"
echo "  - Serial: Thomas algorithm (reference implementation)"
echo "  - Parallel: Schur complement solver"
echo ""
echo "Running with different numbers of MPI processes..."
echo "================================================================"
echo ""

# Test with 1 process (should match serial exactly)
echo "TEST 1: Single process (np=1)"
echo "-----------------------------------------------------------"
mpirun -np 1 ./bin/test_schur_solver | grep -A 20 "Test 8:"
echo ""

# Test with 2 processes
echo "TEST 2: Two processes (np=2)"
echo "-----------------------------------------------------------"
mpirun -np 2 ./bin/test_schur_solver | grep -A 20 "Test 8:"
echo ""

# Test with 4 processes
echo "TEST 3: Four processes (np=4)"
echo "-----------------------------------------------------------"
mpirun -np 4 ./bin/test_schur_solver | grep -A 20 "Test 8:"
echo ""

echo "================================================================"
echo "Comparison complete"
echo "================================================================"
