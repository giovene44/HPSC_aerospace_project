import matplotlib.pyplot as plt
import sys

def plot_speedup(filename):
    procs = []
    time_omp = []

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
            # Read line by line
            for line in lines:
                # Skip empty lines or comments
                if not line.strip() or line.strip().startswith('#'):
                    continue
                
                parts = line.split()
                
                # Look for the header or non-numeric lines to skip
                # Assume the "Procs" column is the 12th (index 11)
                # and "TimeOMP" is the 10th (index 9) based on your file.
                # Check: Nx dx dt nsteps L2... TimeNoOMP TimeOMP ... Procs
                try:
                    # Try converting to float to see if it's a data line
                    p = int(parts[11])       # Column Procs
                    t = float(parts[9])      # Column TimeOMP
                    
                    procs.append(p)
                    time_omp.append(t)
                except (ValueError, IndexError):
                    # If conversion fails, it's probably the header or garbage
                    continue

    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return

    if not procs:
        print("No valid data found in the file.")
        return

    # Calculation of Speedup
    # Assume the first data point (or the one with 1 proc) is the baseline
    # Sort the lists based on the number of procs for safety
    data = sorted(zip(procs, time_omp))
    procs = [x[0] for x in data]
    time_omp = [x[1] for x in data]
    
    base_time = time_omp[0] # Time with 1 processor (or the minimum number found)
    speedup = [base_time / t for t in time_omp]
    ideal_speedup = procs # Ideal linear speedup (y=x)

    # Create the Plot
    plt.figure(figsize=(10, 6))
    
    # Plot Real Speedup
    plt.plot(procs, speedup, marker='o', linestyle='-', linewidth=2, color='blue', label='Real Speedup (MPI)')
    
    # Plot Ideal Speedup
    plt.plot(procs, ideal_speedup, linestyle='--', color='gray', label='Ideal Speedup')

    plt.xlabel('Number of Processors')
    plt.ylabel('Speedup')
    plt.title('MPI Speedup Plot')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(procs) # Show only the ticks of the used processors

    output_file = 'speedup_plot.png'
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_speedup(sys.argv[1])
    else:
        # Default if no arguments are passed
        plot_speedup('Speedup_MPI.dat')