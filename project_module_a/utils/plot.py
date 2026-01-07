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
try:
    data = np.loadtxt(filename, skiprows=1)
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)

x = data[:, 0]
error_0 = data[:, 2]

# Determine x-axis label from filename
xlabel = 'Grid Size (N)' if 'N' in filename else 'Time Step (dt)'

# Create the plot
plt.figure(figsize=(10, 6))

# 1. Plot the actual error data
plt.loglog(x, error_0, 'o-', linewidth=2, markersize=8, label='Error Component 0', zorder=3)

# 2. Plot normalized reference lines starting from the first data point
# We use (x/x[0]) to ensure that at the first point the value is 1.0 * error_0[0]
x0 = x[0]
y0 = error_0[0]

# O(h^-1) reference
plt.loglog(x, y0 * (x / x0)**(-1), 'k--', label='O(h^-1)', alpha=0.7, zorder=1)

# O(h^-2) reference
plt.loglog(x, y0 * (x / x0)**(-2), 'k-.', label='O(h^-2)', alpha=0.7, zorder=2)

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

# --- Convergence Summary Calculation ---

grid_sizes = data[:, 0].astype(int)
dt_values = data[:, 1].astype(float)
errors = data[:, 2].astype(float)
L = data[0, 3].astype(float) if data.shape[1] > 3 else 1.0

# Compute spatial convergence rates
if grid_sizes.size > 1 and grid_sizes[0] != grid_sizes[1]:
    spacial_rates = np.zeros_like(errors)
    for i in range(1, errors.size):
        dx_curr = compute_dx(L, grid_sizes[i])
        dx_prev = compute_dx(L, grid_sizes[i - 1])
        if dx_curr > 0 and dx_prev > 0:
            spacial_rates[i] = np.log(errors[i - 1] / errors[i]) / np.log(dx_prev / dx_curr)
else:
    spacial_rates = np.zeros_like(errors)

# Compute temporal convergence rates
if dt_values.size > 1 and dt_values[0] != dt_values[1]:
    temporal_rates = np.zeros_like(errors)
    for i in range(1, errors.size):
        dt_curr = dt_values[i]
        dt_prev = dt_values[i - 1]
        if dt_curr > 0 and dt_prev > 0:
            temporal_rates[i] = np.log(errors[i - 1] / errors[i]) / np.log(dt_prev / dt_curr)
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
    t_rate = temporal_rates[i]
    s_rate = spacial_rates[i]
    print(f"{grid_sizes[i]:<12} {dx_curr:>12.4e} {dt_curr:>12.4e} {error_curr:>14.6e} {s_rate:>14.4f} {t_rate:>14.4f}")
print("=" * 80)
print(f"Results read from: {filename}")

plt.show()
