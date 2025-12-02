#!/usr/bin/env python3
"""
Simple plotting script for convergence analysis
Usage: python3 plot.py error_vs_N.dat
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python3 plot.py <data_file>")
    print("Example: python3 plot.py error_vs_N.dat")
    sys.exit(1)

# Read the data file
filename = sys.argv[1]
data = np.loadtxt(filename, skiprows=1)

x = data[:, 0]
error_0 = data[:, 2]

# Determine x-axis label from filename
xlabel = 'Grid Size (N)' if 'N' in filename else 'Time Step (dt)'

# Create the plot
plt.figure(figsize=(10, 6))
plt.loglog(x, error_0, 'o-', linewidth=2, markersize=8, label='Error Component 0')
plt.loglog(x, 1/(error_0 * (x / x[0])), 'k--', label='O(h^-1)', alpha=0.7)
plt.loglog(x, 1/(error_0 * (x / x[0])**2), 'k-.', label='O(h^-2)', alpha=0.7)
plt.xlabel(xlabel, fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.title(f'Convergence Analysis: {filename}', fontsize=14)
plt.grid(True, which='both', alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()

# Save and show
output_name = filename.replace('.dat', '.png')
plt.savefig(output_name, dpi=300, bbox_inches='tight')
print(f"Plot saved as {output_name}")
plt.show()

# Print a convergence summary similar to the C printf output
# Try to infer column layout, be tolerant of 2..4+ columns
grid_sizes = data[:, 0].astype(int)
dt_values = data[:, 1].astype(float)
errors = data[:, 2].astype(float)

# Compute convergence rates (log ratio). Use dx_values if positive, else grid_sizes.
rates = np.zeros_like(errors)
for i in range(errors.size):
    dx_curr = 1.0 / grid_sizes[i]
    dx_prev = 1.0 / grid_sizes[i - 1]
    if dx_curr > 0 and dx_prev > 0:
        rates[i] = np.log(errors[i - 1] / errors[i]) / np.log(dx_prev / dx_curr)
    else:
        rates[i] = float('0.0')

# Print formatted summary
print()
print("=" * 49)
print("CONVERGENCE STUDY SUMMARY")
print("=" * 49)
print(f"{'Grid Size':<12} {'dx':>12} {'dt':>12} {'L2 Error':>14} {'Conv. Rate':>12}")
print("-" * 63)
for i in range(errors.size):
    dx_curr = 1.0 / grid_sizes[i]
    dt_curr = dt_values[i]
    error_curr = errors[i]
    rate_curr = rates[i] if i > 0 else 0.0
    print(f"{grid_sizes[i]:<12} {dx_curr:>12.4e} {dt_curr:>12.4e} {error_curr:>14.6e} {rate_curr:>12.4f}")
print("=" * 49)
print(f"Results read from: {filename}")