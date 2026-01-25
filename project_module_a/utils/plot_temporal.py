import matplotlib.pyplot as plt
import sys
import numpy as np

def plot_temporal_convergence(filename):
    dt_vals = []
    l2_u_vals = []
    l2_p_vals = []

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                # Skip header and empty lines
                if not line.strip() or 'Nx' in line or 'source' in line:
                    continue
                
                parts = line.split()
                
                try:
                    # Parsing based on your file:
                    # Col 2: dt (index 2)
                    # Col 4: L2_u_abs (index 4)
                    # Col 5: L2_p_abs (index 5)
                    dt = float(parts[2])
                    l2_u = float(parts[6])
                    l2_p = float(parts[7])
                    
                    dt_vals.append(dt)
                    l2_u_vals.append(l2_u)
                    l2_p_vals.append(l2_p)
                except (ValueError, IndexError):
                    continue
                    
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return

    if not dt_vals:
        print("No valid data found.")
        return

    # Sort data by dt descending (from largest to smallest)
    sorted_data = sorted(zip(dt_vals, l2_u_vals, l2_p_vals), key=lambda x: x[0], reverse=True)
    dt_vals = [x[0] for x in sorted_data]
    l2_u_vals = [x[1] for x in sorted_data]
    l2_p_vals = [x[2] for x in sorted_data]

    # Create Log-Log Plot
    plt.figure(figsize=(10, 7))
    
    # Plot Velocity Error
    plt.loglog(dt_vals, l2_u_vals, 'bo-', linewidth=2, label='Velocity Error ($L^2_{rel}$)')
    
    # Plot Pressure Error
    plt.loglog(dt_vals, l2_p_vals, 'rs-', linewidth=2, label='Pressure Error ($L^2_{rel}$)')

    # --- Reference Lines (Slopes) ---
    # Take the first point (largest dt) as reference
    ref_dt = np.array(dt_vals)
    
    # 1st Order Slope (y ~ x)
    # Calibrate on initial pressure error
    c1 = l2_p_vals[0] / ref_dt[0]
    y_order1 = c1 * ref_dt
    plt.loglog(ref_dt, y_order1, 'k--', label='1st Order Slope ($O(\Delta t)$)')
    
    # 2nd Order Slope (y ~ x^2)
    # Calibrate on initial pressure error (slightly shifted down for visibility)
    c2 = (l2_p_vals[0] * 0.5) / (ref_dt[0]**2)
    y_order2 = c2 * (ref_dt**2)
    plt.loglog(ref_dt, y_order2, 'g:', linewidth=2, label='2nd Order Slope ($O(\Delta t^2)$)')

    # Formatting
    plt.xlabel('Time Step ($\Delta t$)', fontsize=12)
    plt.ylabel('$L^2$ Error', fontsize=12)
    plt.title('Temporal Convergence Analysis', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(fontsize=12)
    
    # Save Figure
    output_filename = 'temporal_convergence.png'
    plt.savefig(output_filename)
    print(f"Plot saved as {output_filename}")
    plt.show()

if __name__ == "__main__":
    # If a file is passed as argument, use it; otherwise use the default
    if len(sys.argv) > 1:
        plot_temporal_convergence(sys.argv[1])
    else:
        plot_temporal_convergence('Convergence_Analysis_2026-01-24_18-15-08_MPI_4_OpenMP_2.dat')