#!/bin/bash

# Display speedup results in a nice table format

if [ $# -lt 1 ]; then
    echo "Usage: $0 <csv_file>"
    exit 1
fi

CSV_FILE=$1

if [ ! -f "$CSV_FILE" ]; then
    echo "Error: File $CSV_FILE not found"
    exit 1
fi

# Extract N from filename
N=$(echo $CSV_FILE | grep -oE 'N[0-9]+' | grep -oE '[0-9]+')

echo "==========================================="
echo "      SPEEDUP ANALYSIS RESULTS"
echo "==========================================="
echo "Problem Size: N = $N"
echo ""
echo "┌───────────┬──────────────┬────────────────┬──────────┬────────────┐"
echo "│ Processes │ Serial Time  │ Parallel Time  │ Speedup  │ Efficiency │"
echo "├───────────┼──────────────┼────────────────┼──────────┼────────────┤"

# Read CSV and format output
tail -n +2 $CSV_FILE | while IFS=, read -r np serial_time parallel_time speedup efficiency; do
    efficiency_pct=$(echo "scale=2; $efficiency * 100" | bc)
    printf "│ %-9s │ %-12s │ %-14s │ %-8s │ %6s%%   │\n" \
           "$np" \
           "${serial_time}s" \
           "${parallel_time}s" \
           "${speedup}x" \
           "$efficiency_pct"
done

echo "└───────────┴──────────────┴────────────────┴──────────┴────────────┘"
echo ""
echo "Key Observations:"
echo "----------------"

# Find best speedup
BEST_SPEEDUP=0
BEST_NP=0
tail -n +2 $CSV_FILE | while IFS=, read -r np serial_time parallel_time speedup efficiency; do
    result=$(echo "$speedup > $BEST_SPEEDUP" | bc)
    if [ "$result" -eq 1 ]; then
        echo "$np,$speedup"
    fi
done | tail -1 | {
    read -r best
    if [ -n "$best" ]; then
        np=$(echo $best | cut -d',' -f1)
        speedup=$(echo $best | cut -d',' -f2)
        echo "- Best speedup: ${speedup}x with $np processes"
    fi
}

echo ""
echo "Notes:"
echo "- Speedup < 1.0 means parallel is slower than serial"
echo "- Ideal speedup would equal number of processes"
echo "- Efficiency = Speedup / Number of Processes"
echo "- 100% efficiency means perfect linear scaling"
echo ""
