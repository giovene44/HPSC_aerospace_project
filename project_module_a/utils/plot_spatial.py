import matplotlib.pyplot as plt
import sys
import numpy as np

def plot_convergence(filename):
    dx_vals = []
    l2_u_vals = []
    l2_p_vals = []

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                # Skip empty lines or headers
                if not line.strip() or 'Nx' in line or 'source' in line:
                    continue
                
                parts = line.split()
                
                try:
                    # Parsing columns based on file structure:
                    # Idx 1: dx
                    # Idx 4: L2_u_abs
                    # Idx 5: L2_p_abs
                    dx = float(parts[1])
                    l2_u = float(parts[4])
                    l2_p = float(parts[5])
                    
                    dx_vals.append(dx)
                    l2_u_vals.append(l2_u)
                    l2_p_vals.append(l2_p)
                except (ValueError, IndexError):
                    continue
                    
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return

    if not dx_vals:
        print("No valid data found.")
        return

    # Sort data by dx (important for line plots)
    # Zip, sort, and unzip
    sorted_data = sorted(zip(dx_vals, l2_u_vals, l2_p_vals), key=lambda x: x[0], reverse=True)
    dx_vals = [x[0] for x in sorted_data]
    l2_u_vals = [x[1] for x in sorted_data]
    l2_p_vals = [x[2] for x in sorted_data]

    # Create Log-Log Plot
    plt.figure(figsize=(10, 7))
    
    # Plot Velocity Error
    plt.loglog(dx_vals, l2_u_vals, 'bo-', linewidth=2, label='Velocity Error ($L^2_{abs}$)')
    
    # Plot Pressure Error
    plt.loglog(dx_vals, l2_p_vals, 'rs-', linewidth=2, label='Pressure Error ($L^2_{abs}$)')

    # Add Reference Slopes (e.g., 2nd Order)
    # We take the last point as reference to position the slope line
    ref_x = [dx_vals[0], dx_vals[-1]]
    
    # 2nd Order Slope (y = C * x^2)
    # Aligning with the first point of Velocity
    c2 = l2_u_vals[0] / (dx_vals[0]**2)
    ref_y2 = [c2 * (x**2) for x in ref_x]
    plt.loglog(ref_x, ref_y2, 'k--', label='2nd Order Slope ($O(\Delta x^2)$)')

    # Formatting
    plt.xlabel('Grid Spacing ($\Delta x$)', fontsize=12)
    plt.ylabel('$L^2$ Norm Error', fontsize=12)
    plt.title('Spatial Convergence Analysis', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(fontsize=12)
    
    # Invert X axis? Usually convergence plots go from large dx to small dx (left to right or right to left)
    # Standard is usually large values on the right. matplotlib does this by default.
    
    output_filename = 'convergence_plot.png'
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_convergence(sys.argv[1])
    else:
        # Default filename if not provided
        plot_convergence('Convergence_Analysis_2026-01-24_16-39-23_MPI_4_OpenMP_2.dat')