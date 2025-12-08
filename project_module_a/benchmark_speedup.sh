#!/bin/bash
# Speedup Benchmark Script for MPI Navier-Stokes Solver

echo "====================================="
echo "MPI Navier-Stokes Speedup Benchmark"
echo "====================================="
echo ""

# Grid sizes to test (use larger grids for meaningful results)
GRID_SIZES=(60 80 100)
PROCESS_COUNTS=(1 2 4)

# Results file
RESULTS_FILE="speedup_results.txt"
echo "Grid_Size,Processes,Time_seconds" > $RESULTS_FILE

for GRID in "${GRID_SIZES[@]}"; do
    echo "Testing grid size: ${GRID}x${GRID}x${GRID} = $((GRID*GRID*GRID)) points"
    echo "-------------------------------"
    
    for NP in "${PROCESS_COUNTS[@]}"; do
        echo "  Running with $NP process(es)..."
        
        # Create temporary input file with desired grid size (line 14: "Nx Ny Nz")
        sed "s/^10 10 10$/${GRID} ${GRID} ${GRID}/" Input/Input.in > Input/Input_temp.in
        cp Input/Input_temp.in Input/Input.in
        
        # Run and time the solver
        START=$(date +%s.%N)
        
        if [ $NP -eq 1 ]; then
            # Serial execution
            ./bin/main_app > /dev/null 2>&1
        else
            # Parallel execution
            mpirun --mca btl self,sm -np $NP ./bin/main_app > /dev/null 2>&1
        fi
        
        END=$(date +%s.%N)
        ELAPSED=$(echo "$END - $START" | bc)
        
        echo "    Time: ${ELAPSED}s"
        echo "${GRID},${NP},${ELAPSED}" >> $RESULTS_FILE
        
        # Restore original input file
        sed "s/^${GRID} ${GRID} ${GRID}$/10 10 10/" Input/Input.in > Input/Input_temp.in
        cp Input/Input_temp.in Input/Input.in
    done
    echo ""
done

# Clean up
rm -f Input/Input_temp.in

echo ""
echo "====================================="
echo "Benchmark Complete!"
echo "====================================="
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "Calculating speedup..."
echo ""

# Calculate and display speedup
uv run python - << 'EOF'
import csv

data = {}
with open('speedup_results.txt', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        grid = int(row['Grid_Size'])
        procs = int(row['Processes'])
        time = float(row['Time_seconds'])
        
        if grid not in data:
            data[grid] = {}
        data[grid][procs] = time

print("Speedup Analysis:")
print("=" * 60)
for grid in sorted(data.keys()):
    print(f"\nGrid: {grid}x{grid}x{grid} ({grid**3} points)")
    print("-" * 60)
    serial_time = data[grid][1]
    
    for procs in sorted(data[grid].keys()):
        parallel_time = data[grid][procs]
        speedup = serial_time / parallel_time
        efficiency = (speedup / procs) * 100
        
        print(f"  {procs} process(es): {parallel_time:.3f}s | "
              f"Speedup: {speedup:.2f}x | Efficiency: {efficiency:.1f}%")

print("\n" + "=" * 60)
EOF

echo ""
echo "To visualize results, run: uv run python plot_speedup.py"

