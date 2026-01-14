#!/bin/bash

# Speedup Benchmark Script
# Compares serial vs parallel performance for 2, 4, 6, 8, 10 processes

set -e

# Problem size (default N=10000 for meaningful timing)
N=${1:-10000}

# Number of solve iterations per measurement (amortizes preprocess cost)
ITERS=${2:-100}

# Number of repetitions for averaging
REPS=5

echo "==========================================="
echo "  Speedup Benchmark: Serial vs Parallel"
echo "==========================================="
echo "Problem size: N = $N"
echo "Solve iterations per measurement: $ITERS"
echo "Repetitions per configuration: $REPS"
echo ""

# Array of process counts to test
NP_VALUES=(2 4 6 8 10)

# Output file for results
RESULTS_FILE="speedup_results_N${N}.txt"
CSV_FILE="speedup_results_N${N}.csv"

# Clean previous results
> $RESULTS_FILE
> $CSV_FILE

# CSV header
echo "np,time_serial_avg,time_parallel_avg,speedup,efficiency" > $CSV_FILE

echo "Building executables..."
# Build dependencies first
make MPI=1 compare > /dev/null 2>&1
# Now build timed versions
g++ -std=c++17 -O2 -Iinc tests/run_serial_solver_timed.cpp -o bin/run_serial_solver_timed 2>&1 | grep -v warning || true
mpicxx -std=c++17 -O2 -DUSE_MPI -Iinc tests/run_parallel_solver_timed.cpp obj/src/navier_stokes_brinkman.o obj/src/run_err.o muparserx/build/libmuparserx.a -o bin/run_parallel_solver_timed 2>&1 | grep -v warning || true
echo "Build complete."
echo ""

# Measure serial baseline
echo "==========================================="
echo "Running Serial Baseline (Thomas Algorithm)"
echo "==========================================="

SERIAL_SUM=0
for i in $(seq 1 $REPS); do
    echo -n "  Run $i/$REPS... "
    TIME=$(./bin/run_serial_solver_timed $N $ITERS)
    SERIAL_SUM=$(echo "$SERIAL_SUM + $TIME" | bc)
    echo "${TIME}s (avg per solve)"
done

# Calculate serial average
SERIAL_AVG=$(echo "scale=6; $SERIAL_SUM / $REPS" | bc)
echo ""
echo "Serial average time: ${SERIAL_AVG}s"
echo ""

# Log to results file
echo "==========================================" >> $RESULTS_FILE
echo "  Speedup Benchmark Results" >> $RESULTS_FILE
echo "==========================================" >> $RESULTS_FILE
echo "Problem size: N = $N" >> $RESULTS_FILE
echo "Solve iterations per measurement: $ITERS" >> $RESULTS_FILE
echo "Repetitions: $REPS" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE
echo "Serial Baseline (Thomas Algorithm):" >> $RESULTS_FILE
echo "  Average time: ${SERIAL_AVG}s" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# Measure parallel for different process counts
echo "==========================================="
echo "Running Parallel Benchmarks (Schur Method)"
echo "==========================================="
echo ""

for np in "${NP_VALUES[@]}"; do
    echo "--- Testing with $np processes ---"

    PARALLEL_SUM=0
    for i in $(seq 1 $REPS); do
        echo -n "  Run $i/$REPS... "
        TIME=$(mpirun -np $np ./bin/run_parallel_solver_timed $N $ITERS 2>/dev/null)
        PARALLEL_SUM=$(echo "$PARALLEL_SUM + $TIME" | bc)
        echo "${TIME}s (avg per solve)"
    done

    # Calculate parallel average
    PARALLEL_AVG=$(echo "scale=6; $PARALLEL_SUM / $REPS" | bc)

    # Calculate speedup and efficiency
    SPEEDUP=$(echo "scale=4; $SERIAL_AVG / $PARALLEL_AVG" | bc)
    EFFICIENCY=$(echo "scale=4; $SPEEDUP / $np" | bc)

    echo ""
    echo "  Average time: ${PARALLEL_AVG}s"
    echo "  Speedup: ${SPEEDUP}x"
    echo "  Efficiency: ${EFFICIENCY} ($(echo "scale=2; $EFFICIENCY * 100" | bc)%)"
    echo ""

    # Log to results file
    echo "Parallel with $np processes:" >> $RESULTS_FILE
    echo "  Average time: ${PARALLEL_AVG}s" >> $RESULTS_FILE
    echo "  Speedup: ${SPEEDUP}x" >> $RESULTS_FILE
    echo "  Efficiency: ${EFFICIENCY} ($(echo "scale=2; $EFFICIENCY * 100" | bc)%)" >> $RESULTS_FILE
    echo "" >> $RESULTS_FILE

    # Write to CSV
    echo "$np,$SERIAL_AVG,$PARALLEL_AVG,$SPEEDUP,$EFFICIENCY" >> $CSV_FILE
done

echo "==========================================="
echo "Benchmark Complete!"
echo "==========================================="
echo "Results saved to:"
echo "  - $RESULTS_FILE (human-readable)"
echo "  - $CSV_FILE (CSV format)"
echo ""

# Create Python plotting script
cat > plot_speedup.py << 'EOF'
#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python3 plot_speedup.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]

# Read data
data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
np_values = data[:, 0]
speedup = data[:, 3]
efficiency = data[:, 4]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Speedup
ax1.plot(np_values, speedup, 'o-', linewidth=2, markersize=8, label='Actual Speedup', color='blue')
ax1.plot(np_values, np_values, '--', linewidth=2, alpha=0.7, label='Ideal Speedup (linear)', color='red')
ax1.set_xlabel('Number of Processes', fontsize=12)
ax1.set_ylabel('Speedup', fontsize=12)
ax1.set_title('Parallel Speedup vs Number of Processes', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_xticks(np_values)

# Plot 2: Efficiency
ax2.plot(np_values, efficiency * 100, 'o-', linewidth=2, markersize=8, color='green', label='Parallel Efficiency')
ax2.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Ideal Efficiency (100%)')
ax2.set_xlabel('Number of Processes', fontsize=12)
ax2.set_ylabel('Efficiency (%)', fontsize=12)
ax2.set_title('Parallel Efficiency vs Number of Processes', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.set_xticks(np_values)
ax2.set_ylim([0, 110])

plt.tight_layout()

# Extract N from filename
import re
match = re.search(r'N(\d+)', csv_file)
N = match.group(1) if match else 'unknown'
output_file = f'speedup_plot_N{N}.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_file}")

# Print summary
print("\n" + "="*50)
print("Speedup Summary:")
print("="*50)
for i, np in enumerate(np_values):
    print(f"  {int(np)} processes: Speedup = {speedup[i]:.3f}x, Efficiency = {efficiency[i]*100:.2f}%")
print("="*50)
EOF

chmod +x plot_speedup.py

echo "To visualize results, run:"
echo "  python3 plot_speedup.py $CSV_FILE"
