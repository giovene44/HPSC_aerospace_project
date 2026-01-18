#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

# Create a seaborn-styled chart
sns.set_theme(style="whitegrid")
fig_seaborn = plt.figure(figsize=(8, 6))

# Prepare data for seaborn
df = pd.DataFrame({
    'Processes': np_values,
    'Speedup': speedup,
})

# Create single plot with seaborn
ax_sns = fig_seaborn.add_subplot(1, 1, 1)

# Plot: Speedup only
sns.lineplot(data=df, x='Processes', y='Speedup', ax=ax_sns, marker='o',
             markersize=10, linewidth=2.5, color='#2E86AB', label='Speedup')
ax_sns.set_xlabel('Number of Processes', fontsize=13, fontweight='bold')
ax_sns.set_ylabel('Speedup', fontsize=13, fontweight='bold')
ax_sns.set_title('Parallel Speedup Analysis', fontsize=15, fontweight='bold', pad=15)
ax_sns.legend(fontsize=11, loc='upper left')
ax_sns.set_xticks(np_values)

plt.tight_layout()
output_file_seaborn = f'speedup_plot_seaborn_N{N}.png'
plt.savefig(output_file_seaborn, dpi=300, bbox_inches='tight')
print(f"Seaborn plot saved to: {output_file_seaborn}")

# Print summary
print("\n" + "="*50)
print("Speedup Summary:")
print("="*50)
for i, np in enumerate(np_values):
    print(f"  {int(np)} processes: Speedup = {speedup[i]:.3f}x, Efficiency = {efficiency[i]*100:.2f}%")
print("="*50)
