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
error_0 = data[:, 1]

# Determine x-axis label from filename
xlabel = 'Grid Size (N)' if 'N' in filename else 'Time Step (dt)'

# Create the plot
plt.figure(figsize=(10, 6))
plt.loglog(x, error_0, 'o-', linewidth=2, markersize=8, label='Error Component 0')

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