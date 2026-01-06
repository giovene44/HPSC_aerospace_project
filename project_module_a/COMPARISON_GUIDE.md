# Serial vs Parallel Solver Comparison Guide

## Overview

Two simple programs have been created to compare the serial Thomas solver with the parallel Schur complement solver:

- `bin/run_serial_solver` - Serial Thomas algorithm
- `bin/run_parallel_solver` - Parallel Schur complement solver

## Building

```bash
# Build both comparison tools
make MPI=1 compare
```

## Running

### Serial Solver
```bash
./bin/run_serial_solver [N]
# Example: ./bin/run_serial_solver 100
# Creates: solution_serial_N100.dat
```

### Parallel Solver
```bash
mpirun -np [NP] ./bin/run_parallel_solver [N]
# Example: mpirun -np 2 ./bin/run_parallel_solver 100
# Creates: solution_parallel_N100_np2.dat
```

## Comparing Solutions

### Manual Comparison
```bash
# Run serial
./bin/run_serial_solver 100

# Run parallel with different process counts
mpirun -np 1 ./bin/run_parallel_solver 100
mpirun -np 2 ./bin/run_parallel_solver 100
mpirun -np 4 ./bin/run_parallel_solver 100

# Compare using awk
awk -f /tmp/compare.awk solution_serial_N100.dat solution_parallel_N100_np1.dat
awk -f /tmp/compare.awk solution_serial_N100.dat solution_parallel_N100_np2.dat
```

Where `/tmp/compare.awk` contains:
```awk
BEGIN {
    max_err = 0
    sum_sq = 0
    count = 0
}
NR == FNR {
    serial[NR] = $2
    next
}
{
    diff = serial[FNR] - $2
    abs_diff = (diff < 0) ? -diff : diff
    if (abs_diff > max_err) max_err = abs_diff
    sum_sq += diff * diff
    count++
}
END {
    l2_err = sqrt(sum_sq)
    printf "Maximum absolute error: %.6e\n", max_err
    printf "L2 norm of error:       %.6e\n", l2_err
    printf "Number of points:       %d\n", count
    if (max_err < 1e-4) {
        print "\n✓ Solutions MATCH (error < 1e-4)"
    } else if (max_err < 1e-2) {
        print "\n⚠ Solutions are CLOSE but with moderate error"
    } else {
        print "\n✗ Solutions DIFFER significantly"
    }
}
```

## Current Results

### Test Problem
- Tridiagonal system: `-u[i-1] + 2*u[i] - u[i+1] = sin(i*π/N)`
- Boundary conditions: a[0] = 0, c[N-1] = 0
- Problem size: N = 100

### Results

**With np=1 (single process):**
```
Maximum absolute error: 0.000000e+00
L2 norm of error:       0.000000e+00
✓ Solutions MATCH (error < 1e-4)
```

**With np=2 (two processes):**
```
Maximum absolute error: 1.029051e+03
L2 norm of error:       5.954100e+03
✗ Solutions DIFFER significantly
```

## Issue Analysis

- ✅ **Single process**: Parallel solver matches serial exactly
- ❌ **Multiple processes**: Solutions differ significantly

This indicates there's an issue with:
1. Domain decomposition with overlapping interfaces
2. Communication of interface values between processes
3. Assembly of the global solution from local partitions

## Solution Files Format

Output files are space-separated with two columns:
```
<index> <solution_value>
0 31.51321220397949
1 63.02642440795898
2 94.50822448730469
...
```

## Quick Comparison Commands

```bash
# Show first 10 values side by side
paste solution_serial_N100.dat solution_parallel_N100_np2.dat | \
  awk '{printf "%3d  Serial: %12.6f  Parallel: %12.6f  Diff: %12.6f\n", \
        $1, $2, $4, $2-$4}' | head -10

# Plot comparison (if gnuplot installed)
gnuplot <<EOF
set terminal png
set output 'comparison.png'
plot 'solution_serial_N100.dat' with lines title 'Serial', \
     'solution_parallel_N100_np2.dat' with points title 'Parallel (np=2)'
EOF
```

## Next Steps

The parallel implementation works correctly for single process but needs debugging for multiple processes. The issue is likely in:

1. **[run_parallel_solver.cpp:503-521](tests/run_parallel_solver.cpp#L503-L521)** - Local coefficient extraction accounting for overlaps
2. **[run_parallel_solver.cpp:102-135](tests/run_parallel_solver.cpp#L102-L135)** - Solution gathering from multiple ranks
3. **[SchurComplementSolver.hpp](inc/SchurComplementSolver.hpp)** - Interface communication and assembly

The domain decomposition logic in `SchurMatrixData::init()` creates overlapping partitions where adjacent processes share interface points. This overlap handling appears to have a bug when distributing coefficients or gathering solutions.
