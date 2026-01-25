import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
DATA_DIR = Path(".")  # folder where the .dat files are
PATTERN = "Convergence_Analysis_MPI_*_OpenMP_*.dat"

# Choose which Nx to use for the runtime plot:
# - set to an integer (e.g., 64) to force that Nx
# - set to None to auto-pick the maximum Nx available per (MPI, OMP) file-set
NX_FOR_TIMING = None


# -----------------------------
# Helpers
# -----------------------------
def parse_mpi_omp_from_name(fname: str):
    """
    Extract MPI and OMP from filenames like:
    Convergence_Analysis_MPI_4_OpenMP_8.dat
    """
    m = re.search(r"MPI_(\d+)_OpenMP_(\d+)", fname)
    if not m:
        raise ValueError(f"Cannot parse MPI/OMP from filename: {fname}")
    return int(m.group(1)), int(m.group(2))


def read_one_file(path: Path) -> pd.DataFrame:
    mpi, omp = parse_mpi_omp_from_name(path.name)

    # read whitespace-separated table with a header line
    df = pd.read_csv(path, sep=r"\s+", engine="python")

    # attach config columns
    df["MPI"] = mpi
    df["OMP"] = omp
    df["TotalCores"] = df["MPI"] * df["OMP"]

    return df


# -----------------------------
# Load all data
# -----------------------------
files = sorted(DATA_DIR.glob(PATTERN))
if not files:
    raise FileNotFoundError(f"No files found matching {PATTERN} in {DATA_DIR.resolve()}")

all_df = pd.concat([read_one_file(f) for f in files], ignore_index=True)

# Ensure numeric (sometimes tabs/spaces can cause object dtype)
for c in all_df.columns:
    if c in {"MPI", "OMP"}:
        continue
    all_df[c] = pd.to_numeric(all_df[c], errors="coerce")

# -----------------------------
# Remove duplicate "1 thread" OpenMP entries
# -----------------------------
# Your rows already contain both TimeNoOMP and TimeOMP.
# The duplication problem typically happens when plotting both:
# - serial baseline (TimeNoOMP) AND
# - OpenMP run with OMP=1 (TimeOMP)
#
# Strategy:
# - Use TimeOMP for all OMP values (including OMP=1)

timing_df = all_df.copy()
timing_df["Time"] = timing_df["TimeOMP"]
timing_plot_df = timing_df.loc[:, ["Nx", "MPI", "OMP", "TotalCores", "Time"]].copy()

# -----------------------------
# Calculate speedup relative to serial (MPI=1, OMP=1)
# -----------------------------
serial_baseline = timing_plot_df[(timing_plot_df["MPI"] == 1) & (timing_plot_df["OMP"] == 1)].copy()
serial_baseline = serial_baseline[["Nx", "Time"]].rename(columns={"Time": "SerialTime"})

# Merge to get serial time for each Nx
speedup_df = timing_plot_df.merge(serial_baseline, on="Nx", how="left")
speedup_df["Speedup"] = speedup_df["SerialTime"] / speedup_df["Time"]

# -----------------------------
# Plot both runtime and speedup
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Runtime vs Nx
ax = axes[0]
for (mpi, omp), g in timing_plot_df.groupby(["MPI", "OMP"], sort=True):
    g = g.sort_values("Nx")
    ax.loglog(g["Nx"], g["Time"], marker="o", linewidth=2, label=f"MPI={mpi}, OMP={omp}")

ax.set_xlabel("Nx")
ax.set_ylabel("Runtime [s]")
ax.set_title("Runtime scaling")
ax.grid(True, which="both", linestyle=":")
ax.legend()

# Plot 2: Speedup vs Nx
ax = axes[1]
for (mpi, omp), g in speedup_df.groupby(["MPI", "OMP"], sort=True):
    g = g.sort_values("Nx")
    ax.semilogx(g["Nx"], g["Speedup"], marker="o", linewidth=2, label=f"MPI={mpi}, OMP={omp}")

ax.set_xlabel("Nx")
ax.set_ylabel("Speedup (relative to MPI=1, OMP=1)")
ax.set_title("Speedup relative to serial")
ax.grid(True, which="both", linestyle=":")
ax.legend()

plt.tight_layout()
plt.show()