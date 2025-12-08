#!/bin/bash
GRID=$1
NP=$2

echo "Grid: ${GRID}x${GRID}x${GRID}, Processes: ${NP}"

# Update grid
sed -i.bak "s/^[0-9]* [0-9]* [0-9]*$/${GRID} ${GRID} ${GRID}/" Input/Input.in

# Time it
/usr/bin/time -p mpirun --mca btl self,sm -np ${NP} ./bin/main_app > /dev/null 2>&1

# Restore
sed -i.bak "s/^${GRID} ${GRID} ${GRID}$/10 10 10/" Input/Input.in
