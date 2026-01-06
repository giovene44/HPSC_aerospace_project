#!/usr/bin/env python3
"""
Simple plotting script for convergence analysis
Usage: python3 plot.py error_vs_N.dat
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

def compute_dx(L, N):
    """Compute grid spacing dx given grid size N."""
    return (L) / (N - 0.5)

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
plt.loglog(x, 1/(x**(1)), 'k--', label='O(h^-1)', alpha=0.7)
plt.loglog(x, 1/(x**(2)), 'k-.', label='O(h^-2)', alpha=0.7)
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
L=data[0,3].astype(float) if data.shape[1]>3 else 1.0

# Compute convergence rates (log ratio). Use dx_values if positive, else grid_sizes.
if grid_sizes[0]!=grid_sizes[1]:
    spacial_rates = np.zeros_like(errors)
    for i in range(errors.size):
        dx_curr = compute_dx(L, grid_sizes[i])
        dx_prev = compute_dx(L, grid_sizes[i - 1])
        if dx_curr > 0 and dx_prev > 0:
            spacial_rates[i] = np.log(errors[i - 1] / errors[i]) / np.log(dx_prev / dx_curr)
        else:
            spacial_rates[i] = float('0.0')
else:
    spacial_rates = np.zeros_like(errors)
# Compute convergence rates (log ratio). Use dt_values since dt is halved each iteration.
if dt_values[0]!=dt_values[1]:
    temporal_rates = np.zeros_like(errors)
    for i in range(errors.size):
        dt_curr = dt_values[i]
        dt_prev = dt_values[i - 1]
        if dt_curr > 0 and dt_prev > 0:
            temporal_rates[i] = np.log(errors[i - 1] / errors[i]) / np.log(dt_prev / dt_curr)
        else:
            temporal_rates[i] = 0.0
else:
    temporal_rates = np.zeros_like(errors)

# Print formatted summary
print()
print("=" * 80)
print("CONVERGENCE STUDY SUMMARY")
print("=" * 80)
print(f"{'Grid Size':<12} {'dx':>12} {'dt':>12} {'L2 Error':>14} {'Spacial Rate':>14} {'Temporal Rate':>14}")
print("-" * 80)
for i in range(errors.size):
    dx_curr = compute_dx(L, grid_sizes[i])
    dt_curr = dt_values[i]
    error_curr = errors[i]
    temporal_rate_curr = temporal_rates[i] if i > 0 else 0.0
    spacial_rate_curr = spacial_rates[i] if i > 0 else 0.0
    print(f"{grid_sizes[i]:<12} {dx_curr:>12.4e} {dt_curr:>12.4e} {error_curr:>14.6e} {spacial_rate_curr:>14.4f} {temporal_rate_curr:>14.4f}")
print("=" * 49)
print(f"Results read from: {filename}")