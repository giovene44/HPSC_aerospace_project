#!/usr/bin/env -S uv run python
"""Plot speedup results from benchmark"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read results
df = pd.read_csv('speedup_results.txt')

# Get unique grid sizes
grid_sizes = df['Grid_Size'].unique()

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Speedup curves
ax1 = axes[0]
for grid in sorted(grid_sizes):
    subset = df[df['Grid_Size'] == grid]
    serial_time = subset[subset['Processes'] == 1]['Time_seconds'].values[0]
    
    speedup = serial_time / subset['Time_seconds']
    
    ax1.plot(subset['Processes'], speedup, 'o-', 
             label=f'{grid}³ ({grid**3:,} points)', linewidth=2, markersize=8)

# Add ideal speedup line
max_procs = df['Processes'].max()
ax1.plot([1, max_procs], [1, max_procs], 'k--', alpha=0.5, label='Ideal (linear)')

ax1.set_xlabel('Number of Processes', fontsize=12)
ax1.set_ylabel('Speedup', fontsize=12)
ax1.set_title('Parallel Speedup vs Number of Processes', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(df['Processes'].unique())

# Plot 2: Efficiency
ax2 = axes[1]
for grid in sorted(grid_sizes):
    subset = df[df['Grid_Size'] == grid]
    serial_time = subset[subset['Processes'] == 1]['Time_seconds'].values[0]
    
    speedup = serial_time / subset['Time_seconds']
    efficiency = (speedup / subset['Processes']) * 100
    
    ax2.plot(subset['Processes'], efficiency, 'o-', 
             label=f'{grid}³ ({grid**3:,} points)', linewidth=2, markersize=8)

ax2.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='100% Efficiency')
ax2.set_xlabel('Number of Processes', fontsize=12)
ax2.set_ylabel('Parallel Efficiency (%)', fontsize=12)
ax2.set_title('Parallel Efficiency vs Number of Processes', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(df['Processes'].unique())
ax2.set_ylim([0, 110])

plt.tight_layout()
plt.savefig('speedup_analysis.png', dpi=150, bbox_inches='tight')
print(f"Speedup plot saved to: speedup_analysis.png")
plt.show()

