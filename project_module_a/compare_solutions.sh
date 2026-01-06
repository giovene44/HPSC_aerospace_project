#!/bin/bash
# Script to compare serial and parallel solver solutions

N=${1:-100}  # Default problem size 100
NP=${2:-1}   # Default 1 process

echo "================================================================"
echo "Comparing Serial vs Parallel Solver Solutions"
echo "================================================================"
echo "Problem size: N = $N"
echo "MPI processes: $NP"
echo ""

# Run serial solver
echo "Running serial solver..."
./bin/run_serial_solver $N
echo ""

# Run parallel solver
echo "Running parallel solver with $NP processes..."
mpirun -np $NP ./bin/run_parallel_solver $N
echo ""

# Compare solutions
SERIAL_FILE="solution_serial_N${N}.dat"
PARALLEL_FILE="solution_parallel_N${N}_np${NP}.dat"

if [ -f "$SERIAL_FILE" ] && [ -f "$PARALLEL_FILE" ]; then
    echo "================================================================"
    echo "Computing difference..."
    echo "================================================================"

    # Create Python script to compute differences
    python3 - <<EOF
import numpy as np

# Load solutions
serial = np.loadtxt('$SERIAL_FILE')
parallel = np.loadtxt('$PARALLEL_FILE')

# Extract solution vectors
x_serial = serial[:, 1]
x_parallel = parallel[:, 1]

# Compute errors
diff = x_serial - x_parallel
max_err = np.max(np.abs(diff))
l2_err = np.linalg.norm(diff)
rel_err = l2_err / np.linalg.norm(x_serial) if np.linalg.norm(x_serial) > 0 else 0

print(f"Maximum absolute error: {max_err:.6e}")
print(f"L2 norm of error:       {l2_err:.6e}")
print(f"Relative L2 error:      {rel_err:.6e}")
print()

# Show first few values
print("First 10 values comparison:")
print("i      Serial          Parallel        Difference")
print("-" * 60)
for i in range(min(10, len(x_serial))):
    print(f"{i:3d}    {x_serial[i]:14.6e}  {x_parallel[i]:14.6e}  {diff[i]:14.6e}")

# Check if solutions match
if max_err < 1e-4:
    print()
    print("✓ Solutions MATCH (error < 1e-4)")
elif max_err < 1e-2:
    print()
    print("⚠ Solutions are CLOSE but with moderate error")
else:
    print()
    print("✗ Solutions DIFFER significantly")
EOF
else
    echo "Error: Solution files not found"
    echo "Expected: $SERIAL_FILE and $PARALLEL_FILE"
fi

echo ""
echo "================================================================"
echo "Solution files:"
echo "  Serial:   $SERIAL_FILE"
echo "  Parallel: $PARALLEL_FILE"
echo "================================================================"
