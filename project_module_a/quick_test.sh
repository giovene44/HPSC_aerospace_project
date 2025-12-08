#!/bin/bash
# Quick speedup test with a specific grid size

if [ $# -lt 1 ]; then
    echo "Usage: ./quick_test.sh GRID_SIZE [NUM_PROCESSES]"
    echo "Example: ./quick_test.sh 60 4"
    exit 1
fi

GRID=$1
NP=${2:-4}

echo "====================================="
echo "Quick Speedup Test: ${GRID}³ grid"
echo "====================================="

# Backup original input
cp Input/Input.in Input/Input.in.bak

# Update grid size in Input.in
sed -i.tmp "s/^10 10 10$/${GRID} ${GRID} ${GRID}/" Input/Input.in

echo ""
echo "1. Serial execution (1 process)..."
START=$(date +%s.%N)
mpirun --mca btl self,sm -np 1 ./bin/main_app > /dev/null 2>&1
END=$(date +%s.%N)
SERIAL_TIME=$(echo "$END - $START" | bc)
echo "   Serial time: ${SERIAL_TIME}s"

echo ""
echo "2. Parallel execution ($NP processes)..."
START=$(date +%s.%N)
mpirun --mca btl self,sm -np $NP ./bin/main_app > /dev/null 2>&1
END=$(date +%s.%N)
PARALLEL_TIME=$(echo "$END - $START" | bc)
echo "   Parallel time: ${PARALLEL_TIME}s"

# Calculate speedup
SPEEDUP=$(echo "scale=2; $SERIAL_TIME / $PARALLEL_TIME" | bc)
EFFICIENCY=$(echo "scale=1; 100 * $SPEEDUP / $NP" | bc)

echo ""
echo "====================================="
echo "Results for ${GRID}³ grid ($((GRID*GRID*GRID)) points):"
echo "  Serial:     ${SERIAL_TIME}s"
echo "  Parallel:   ${PARALLEL_TIME}s"
echo "  Speedup:    ${SPEEDUP}x"
echo "  Efficiency: ${EFFICIENCY}%"
echo "====================================="

# Restore original input
mv Input/Input.in.bak Input/Input.in
rm -f Input/Input.in.tmp
