# Parallel Tridiagonal Solver Implementation

## Table of Contents
1. [Overview](#overview)
2. [Implementation Comparison](#implementation-comparison)
3. [Parallel Algorithm: Schur Complement Method](#parallel-algorithm-schur-complement-method)
4. [Architectural Design Choices](#architectural-design-choices)
5. [Code Structure](#code-structure)
6. [Performance Considerations](#performance-considerations)
7. [Tutorial: Running the Solvers](#tutorial-running-the-solvers)

---

## Overview

This project implements two versions of a tridiagonal linear system solver for solving partial differential equations (PDEs) using implicit time-stepping methods:

- **Serial Version**: Classical Thomas Algorithm (tridiagonal matrix algorithm)
- **Parallel Version**: Schur Complement Domain Decomposition with MPI

Both solvers solve systems of the form:

```
a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = rhs[i]
```

The parallel implementation enables solving large-scale problems across multiple processors while maintaining numerical accuracy equivalent to the serial version.

### Key Features

- **MPI-based parallelization** using domain decomposition
- **Schur complement method** for efficient parallel tridiagonal solves
- **Preprocessing/runtime separation** for time-stepping applications
- **Optional OpenMP** support for additional parallelism within each MPI process
- **Scalable architecture** supporting arbitrary number of processes

---

## Implementation Comparison

### Serial Implementation (Thomas Algorithm)

**File**: [`tests/run_serial_solver_timed.cpp`](tests/run_serial_solver_timed.cpp)

The serial solver uses the classical Thomas algorithm, a specialized Gaussian elimination method for tridiagonal systems.

#### Algorithm Steps:

1. **Forward Sweep** - Eliminate lower diagonal:
   ```cpp
   c_prime[0] = c[0] / b[0];
   d_prime[0] = rhs[0] / b[0];

   for (i = 1; i < n; i++) {
       Real denom = b[i] - a[i] * c_prime[i-1];
       c_prime[i] = c[i] / denom;
       d_prime[i] = (rhs[i] - a[i] * d_prime[i-1]) / denom;
   }
   ```

2. **Back Substitution** - Solve for solution:
   ```cpp
   x[n-1] = d_prime[n-1];
   for (i = n-2; i >= 0; i--) {
       x[i] = d_prime[i] - c_prime[i] * x[i+1];
   }
   ```

**Characteristics**:
- ✅ Simple and efficient for single processor
- ✅ O(n) time complexity
- ✅ Minimal memory overhead
- ❌ Inherently sequential (cannot parallelize effectively)
- ❌ Poor scalability for large systems

---

### Parallel Implementation (Schur Complement)

**Files**:
- [`tests/run_parallel_solver_timed.cpp`](tests/run_parallel_solver_timed.cpp)
- [`inc/SchurComplementSolver.hpp`](inc/SchurComplementSolver.hpp)
- [`inc/MPICommunicator.hpp`](inc/MPICommunicator.hpp)
- [`inc/SchurMatrixData.hpp`](inc/SchurMatrixData.hpp)

The parallel solver implements a **Schur complement domain decomposition** strategy that partitions the problem across MPI processes.

#### High-Level Algorithm:

Given a global tridiagonal system, we:

1. **Partition** the domain across `P` processes
2. **Reorder** unknowns into internal points (u_i) and interface points (u_s)
3. **Form block system**:
   ```
   [A_ii  A_is] [u_i]   [f_i]
   [A_si  A_ss] [u_s] = [f_s]
   ```

4. **Eliminate internal unknowns** to form the Schur complement:
   ```
   S = A_ss - A_si * A_ii^(-1) * A_is
   ```

5. **Solve reduced system** for interface values (small system on `P-1` interfaces)

6. **Back-substitute** to recover internal values on each process

**Characteristics**:
- ✅ Fully parallelizable
- ✅ Good scalability for large problems
- ✅ Amortized preprocessing cost (beneficial for time-stepping)
- ✅ Reduced communication (only interface values exchanged)
- ⚠️ Preprocessing overhead for small problems
- ⚠️ Requires careful interface handling

---

## Parallel Algorithm: Schur Complement Method

### Mathematical Foundation

For a tridiagonal system partitioned across processes, each process owns a local block:

```
Process 0: [b₀  c₀  0   ...        ] [x₀]   [f₀]
           [a₁  b₁  c₁  0   ...    ] [x₁]   [f₁]
           [0   a₂  b₂  c₂  ...    ] [x₂] = [f₂]
           ...
Process 1:             [aₖ  bₖ  cₖ ...] [xₖ]   [fₖ]
           ...
```

The **interface points** are the boundary values shared between adjacent processes.

### Implementation Phases

#### Phase 1: Preprocessing (One-time, or when coefficients change)

**Location**: [`SchurComplementSolver::preprocess()`](inc/SchurComplementSolver.hpp#L192-L248)

1. **Extract local blocks**:
   - Identify internal points (excluding interfaces)
   - Store interface coupling coefficients

2. **Compute A_ii^(-1) columns** needed for Schur complement:
   ```cpp
   // Solve A_ii * col = e_j for interface-adjacent columns
   // Left interface: couples to first internal point
   // Right interface: couples to last internal point
   ```

3. **Compute local Schur contributions**:
   ```cpp
   // Diagonal: S[i,i] = b[interface] - coupling * inv_col
   schur_diag_left = -c_left * inv_Aii_col_left[0];
   schur_diag_right = b_right - a_right * inv_Aii_col_right[n-1];

   // Off-diagonal: couplings to neighbor interfaces
   schur_offdiag_to_right = -c_left * inv_Aii_col_right[0];
   ```

4. **Assemble global Schur system** via MPI Allgather:
   ```cpp
   // Each interface gets contributions from two adjacent processes
   S[i] = contribution_from_process[i] + contribution_from_process[i+1]
   ```

**Key Insight**: This preprocessing is done **once** and reused for multiple solves with the same coefficients (common in time-stepping PDEs).

---

#### Phase 2: Runtime Solve (Every timestep)

**Location**: [`SchurComplementSolver::solve()`](inc/SchurComplementSolver.hpp#L408-L555)

1. **Solve local internal systems**:
   ```cpp
   // Each process solves: A_ii * y = f_i
   // Using Thomas algorithm on internal block
   y = thomas_solve(A_ii, f_i);
   ```

2. **Compute modified interface RHS**:
   ```cpp
   // f_s_modified = f_s - A_si * y
   // Process with left interface: contribution = -c[0] * y[0]
   // Process with right interface: contribution = f_s[right] - a[N-1] * y[n-1]
   ```

3. **Reduce interface RHS** (MPI Allreduce):
   ```cpp
   // All processes sum their interface contributions
   MPI_Allreduce(local_interface_rhs, global_interface_rhs, SUM);
   ```

4. **Solve reduced interface system** (redundantly on all processes):
   ```cpp
   // Tiny system (size = P-1), solved using Thomas algorithm
   // S * u_s = f_s_modified
   interface_solution = thomas_solve(S, f_s_modified);
   ```

5. **Back-substitute for internal solution**:
   ```cpp
   // u_i = A_ii^(-1) * (f_i - A_is * u_s)
   g = f_i - A_is * u_s;  // Modify RHS with interface values
   u_i = thomas_solve(A_ii, g);
   ```

**Key Optimization**: Using `MPI_Allreduce` instead of `Reduce + Broadcast` allows all processes to solve the tiny interface system locally, avoiding an extra communication step.

---

### Domain Decomposition Strategy

**Location**: [`SchurComplementData::init()`](inc/SchurMatrixData.hpp#L127-L194)

The domain is partitioned with **overlapping interfaces**:

```
Global indices:  0  1  2  3  4  5  6  7  8  9
                 [--------Process 0---------]
                                [----Process 1----]
                                            [Process 2]

Process 0 owns:  0  1  2  3  4  (right interface at 4)
Process 1 owns:  4  5  6  7     (left interface at 4, right at 7)
Process 2 owns:  7  8  9        (left interface at 7)
```

**Interface ownership**:
- Interface between processes `i` and `i+1` is **shared** by both
- Process `i` stores it as "right interface"
- Process `i+1` stores it as "left interface"

This overlapping strategy:
- ✅ Simplifies local indexing
- ✅ Reduces communication (interfaces are local)
- ✅ Maintains data locality

---

## Architectural Design Choices

### 1. Preprocessing vs Runtime Separation

**Rationale**: Time-stepping PDE solvers repeatedly solve systems with **constant coefficients** but varying RHS.

**Design**:
```cpp
// ONCE at initialization:
schur_solver.preprocess(a, b, c);  // Compute Schur complement

// MANY TIMES per timestep:
for (int t = 0; t < num_timesteps; t++) {
    schur_solver.solve(rhs, solution);  // Just solve
}
```

**Benefits**:
- Amortizes expensive Schur complement computation
- Reduces per-solve cost to O(n_local) + small communication
- Matches typical PDE solver workflow

---

### 2. MPI Communication Strategy

**File**: [`inc/MPICommunicator.hpp`](inc/MPICommunicator.hpp)

**Design Choices**:

1. **Allgather for Schur assembly** ([`line 484`](inc/MPICommunicator.hpp#L484)):
   ```cpp
   MPI_Allgather(local_contrib, global_contrib, ...);
   ```
   - Small data (4 values per process)
   - All processes need the result
   - More efficient than Gather + Broadcast

2. **Allreduce for interface RHS** ([`line 509`](inc/MPICommunicator.hpp#L509)):
   ```cpp
   MPI_Allreduce(local_rhs, global_rhs, MPI_SUM, ...);
   ```
   - Interface system is tiny (P-1 values)
   - All processes solve it redundantly
   - Avoids extra Broadcast after Reduce

3. **Cartesian topology support** ([`line 305`](inc/MPICommunicator.hpp#L305)):
   ```cpp
   MPI_Cart_create(comm, 3, dims, periods, reorder, &cart_comm);
   ```
   - Enables 3D domain decomposition
   - Simplifies neighbor finding
   - Extensible to multi-dimensional PDEs

**Rationale**: Minimize number of collective operations while keeping code simple and readable.

---

### 3. Abstraction Layers

The implementation uses clean separation of concerns:

```
┌─────────────────────────────────────┐
│      Application / Test Code        │
├─────────────────────────────────────┤
│    SchurComplementSolver            │  ← High-level parallel solver
│    - preprocess(), solve()          │
├─────────────────────────────────────┤
│    MPICommunicator                  │  ← MPI wrapper
│    - allreduce, allgather, etc.     │
├─────────────────────────────────────┤
│    SchurMatrixData                  │  ← Data structures
│    - topology, Schur matrices       │
└─────────────────────────────────────┘
```

**Benefits**:
- ✅ Testable components
- ✅ Serial fallback (USE_MPI not defined)
- ✅ Easy to extend (add new communication patterns)
- ✅ Clear API boundaries

---

### 4. Hybrid MPI + OpenMP Support

**File**: [`inc/Solver.hpp`](inc/Solver.hpp#L519-L676)

The velocity solver includes an optional `use_omp` parameter:

```cpp
#ifdef _OPENMP
    if (use_omp) {
        #pragma omp parallel for collapse(2)
        for (Dim i2 = 0; i2 < Outer2; ++i2)
            for (Dim i1 = 0; i1 < Outer1; ++i1)
                worker(i1, i2);  // Solve independent 1D lines
    }
#endif
```

**Design Rationale**:
- **MPI**: Distributes along solve direction (e.g., X-direction)
- **OpenMP**: Parallelizes independent lines (Y-Z plane)
- **Hybrid scaling**: Can exploit both distributed and shared memory

**When to use**:
- MPI alone: Good network, many nodes
- MPI+OpenMP: Multi-core nodes, reduce MPI overhead

---

### 5. Error Handling and Validation

**Numerical stability checks**:

```cpp
// Thomas algorithm pivot check
if (std::abs(denom) < 1e-15) {
    throw std::runtime_error("Thomas algorithm: zero pivot encountered");
}
```

**Size validation**:

```cpp
if (a.size() != local_N || b.size() != local_N || c.size() != local_N) {
    throw std::runtime_error("coefficient size mismatch");
}
```

**Preprocessing guard**:

```cpp
if (!schur_data_.is_preprocessed) {
    throw std::runtime_error("must call preprocess() first");
}
```

---

## Code Structure

### Core Components

#### 1. SchurComplementSolver Class

**Purpose**: Main parallel solver implementing Schur complement method

**Key Methods**:
- `preprocess(a, b, c)`: Compute Schur complement (one-time)
- `solve(rhs, solution)`: Solve system (per timestep)
- `thomas_solve()`: Serial tridiagonal solver (internal use)

**Location**: [`inc/SchurComplementSolver.hpp`](inc/SchurComplementSolver.hpp#L33-L173)

---

#### 2. SchurMatrixData Structures

**Purpose**: Store domain decomposition and Schur complement data

**Components**:

- **SchurBlockData** ([`line 20`](inc/SchurMatrixData.hpp#L20)):
  - Local internal/interface blocks
  - Precomputed inverse columns
  - Schur contributions

- **SchurComplementData** ([`line 70`](inc/SchurMatrixData.hpp#L70)):
  - Process topology (rank, size, neighbors)
  - Global/local indices
  - Assembled Schur system

**Location**: [`inc/SchurMatrixData.hpp`](inc/SchurMatrixData.hpp)

---

#### 3. MPICommunicator Class

**Purpose**: MPI abstraction layer with serial fallback

**Key Features**:
- Initialization/finalization
- Point-to-point communication
- Collective operations (Allgather, Allreduce, Broadcast)
- Cartesian topology management

**Serial Mode**: When `USE_MPI` is not defined, all methods become no-ops, enabling serial execution.

**Location**: [`inc/MPICommunicator.hpp`](inc/MPICommunicator.hpp#L23-L251)

---

#### 4. Integration with Velocity Solver

**Purpose**: Apply parallel solver to 3D PDE problems

**Method**: `block_solver_parallel()` ([`line 519`](inc/Solver.hpp#L519))

**Workflow**:
1. For each independent 1D line (in Y-Z plane):
   - Setup local tridiagonal coefficients
   - Apply boundary conditions
   - Call Schur solver for each velocity component
2. Optional OpenMP parallelization over lines

**Location**: [`inc/Solver.hpp`](inc/Solver.hpp#L519-L676)

---

### File Organization

```
project_module_a/
├── inc/
│   ├── SchurComplementSolver.hpp    # Parallel solver (header-only)
│   ├── SchurMatrixData.hpp          # Data structures
│   ├── MPICommunicator.hpp          # MPI wrapper (header-only)
│   ├── Solver.hpp                   # PDE solver integration
│   └── Variables.hpp                # Type definitions
├── tests/
│   ├── run_serial_solver_timed.cpp  # Serial benchmark
│   ├── run_parallel_solver_timed.cpp # Parallel benchmark
│   └── test_schur_solver.cpp        # Unit tests
├── Makefile                         # Build system
└── benchmark_speedup_final.sh       # Performance analysis script
```

---

## Performance Considerations

### Scalability Analysis

**Strong Scaling** (fixed problem size, increase processes):

- **Serial**: O(N) time, constant
- **Parallel**: O(N/P) + O(P) communication
  - Local solve: O(N/P) per process
  - Interface system: O(P) (very small)
  - Communication: O(log P) for collectives

**Weak Scaling** (problem size grows with processes):

- Ideal: constant time per process
- Schur method: Near-ideal for large N/P ratio

---

### When to Use Parallel Solver

**Use Parallel When**:
- ✅ Problem size N > 10,000
- ✅ Multiple timesteps (amortize preprocessing)
- ✅ Distributed memory system available
- ✅ Strong scaling needed

**Use Serial When**:
- ✅ Small problems (N < 1,000)
- ✅ Single solve (no time-stepping)
- ✅ Single-node execution
- ✅ Simplicity preferred

---

### Optimization Techniques

1. **Cache Efficiency**:
   - Contiguous memory for tridiagonal coefficients
   - Local data access in Thomas algorithm

2. **Communication Overlap** (potential improvement):
   - Could use non-blocking collectives (MPI_Iallreduce)
   - Overlap computation and communication

3. **Load Balancing**:
   - Domain partitioning accounts for remainder: `global_N % num_procs`
   - Ensures balanced work distribution

4. **Memory Layout**:
   - Structure-of-arrays for better vectorization
   - Aligned allocations for SIMD

---

## Tutorial: Running the Solvers

### Prerequisites

**Required**:
- C++17 compatible compiler (g++ 7.0+)
- MPI implementation (OpenMPI, MPICH, Intel MPI)
- CMake 3.10+ (for muparserx dependency)
- GNU Make

**Optional**:
- OpenMP support (for hybrid parallelism)
- Python 3 + matplotlib (for visualization)

---

### Building the Solvers

#### 1. Build Serial Version

```bash
cd project_module_a

# Build serial solver (no MPI, no OpenMP)
make clean
make bin/run_serial_solver_timed
```

**Output**: `bin/run_serial_solver_timed`

---

#### 2. Build Parallel Version

```bash
# Build parallel solver with MPI
make clean
make MPI=1 bin/run_parallel_solver_timed
```

**Output**: `bin/run_parallel_solver_timed`

---

#### 3. Build Both (Using Benchmark Script)

The benchmark script automatically builds both versions:

```bash
./benchmark_speedup_final.sh
```

---

### Running the Solvers

#### Serial Solver

**Syntax**:
```bash
./bin/run_serial_solver_timed <N> <iterations>
```

**Parameters**:
- `N`: Problem size (number of grid points)
- `iterations`: Number of solve iterations to average

**Example**:
```bash
# Solve N=10000 system, 100 iterations
./bin/run_serial_solver_timed 10000 100
```

**Output**:
```
0.000234
```
(Average time per solve in seconds)

---

#### Parallel Solver

**Syntax**:
```bash
mpirun -np <num_procs> ./bin/run_parallel_solver_timed <N> <iterations>
```

**Parameters**:
- `num_procs`: Number of MPI processes
- `N`: Problem size
- `iterations`: Number of solve iterations

**Example**:
```bash
# Run with 4 processes, N=10000, 100 iterations
mpirun -np 4 ./bin/run_parallel_solver_timed 10000 100
```

**Output**:
```
0.000087
```
(Average time per solve in seconds, from rank 0)

---

### Performance Benchmarking

#### Running Complete Benchmark Suite

**Script**: `benchmark_speedup_final.sh`

```bash
# Run with default settings (N=10000, 100 iterations)
./benchmark_speedup_final.sh

# Run with custom problem size
./benchmark_speedup_final.sh 50000

# Run with custom problem size and iterations
./benchmark_speedup_final.sh 50000 200
```

**What it does**:
1. Builds both serial and parallel solvers
2. Runs serial baseline (5 repetitions)
3. Runs parallel with 2, 4, 6, 8, 10 processes (5 reps each)
4. Computes speedup and efficiency
5. Generates CSV and text result files
6. Creates Python plotting script

**Output Files**:
- `speedup_results_N<N>.txt`: Human-readable results
- `speedup_results_N<N>.csv`: CSV data for plotting
- `plot_speedup.py`: Python script for visualization

---

#### Visualizing Results

```bash
# Run benchmark
./benchmark_speedup_final.sh 10000

# Generate plots
python3 plot_speedup.py speedup_results_N10000.csv
```

**Output**: `speedup_plot_N10000.png` with speedup and efficiency curves

---

#### Displaying Results Table

```bash
# Show formatted table
./show_speedup.sh speedup_results_N10000.csv
```

**Example Output**:
```
===========================================
      SPEEDUP ANALYSIS RESULTS
===========================================
Problem Size: N = 10000

┌───────────┬──────────────┬────────────────┬──────────┬────────────┐
│ Processes │ Serial Time  │ Parallel Time  │ Speedup  │ Efficiency │
├───────────┼──────────────┼────────────────┼──────────┼────────────┤
│ 2         │ 0.000234s    │ 0.000156s      │ 1.50x    │  75.00%   │
│ 4         │ 0.000234s    │ 0.000087s      │ 2.69x    │  67.25%   │
│ 6         │ 0.000234s    │ 0.000062s      │ 3.77x    │  62.83%   │
│ 8         │ 0.000234s    │ 0.000051s      │ 4.59x    │  57.38%   │
│ 10        │ 0.000234s    │ 0.000045s      │ 5.20x    │  52.00%   │
└───────────┴──────────────┴────────────────┴──────────┴────────────┘
```

---

### Advanced Usage

#### Hybrid MPI + OpenMP

To enable both MPI and OpenMP:

```bash
# Build with both flags
make clean
make MPI=1 OMP=1 bin/run_parallel_solver_timed

# Run with 2 MPI processes, 4 OpenMP threads each
export OMP_NUM_THREADS=4
mpirun -np 2 ./bin/run_parallel_solver_timed 10000 100
```

---

#### Testing Correctness

Verify parallel solver produces correct results:

```bash
# Build test suite
make MPI=1 test

# Run Schur solver test
./bin/test_schur_solver
```

---

#### Custom Compiler Settings

```bash
# Debug build
make BUILD=debug MPI=1 bin/run_parallel_solver_timed

# Optimized build with specific flags
make MPI=1 CXXFLAGS="-std=c++17 -O3 -march=native -DUSE_MPI" \
     bin/run_parallel_solver_timed
```

---

### Troubleshooting

#### Problem: "MPI not found"

**Solution**: Install MPI implementation
```bash
# Ubuntu/Debian
sudo apt-get install mpich libmpich-dev

# macOS
brew install mpich

# Verify installation
which mpirun
which mpicxx
```

---

#### Problem: Poor scaling

**Possible Causes**:
1. Problem size too small (N < 1000 per process)
2. Network overhead dominates
3. Imbalanced load

**Solutions**:
- Increase problem size: Try N > 10,000
- Use fewer processes for small problems
- Check MPI process binding: `mpirun --bind-to core`

---

#### Problem: "zero pivot encountered"

**Cause**: Ill-conditioned matrix (singular or near-singular)

**Solution**: Check boundary conditions and matrix coefficients

---

### Performance Tips

1. **Problem Size**: Use N ≥ 10,000 for meaningful parallel performance
2. **Iterations**: Use ≥ 100 iterations to amortize preprocessing cost
3. **Process Count**: Best scaling typically with 2-8 processes for N ~ 10,000
4. **Network**: Use InfiniBand or fast network for best scaling
5. **Binding**: Bind MPI processes to cores for consistent performance

---

## Conclusion

This implementation demonstrates a production-quality parallel tridiagonal solver using the Schur complement method. Key achievements:

- ✅ **Scalable**: Efficient for large problems across multiple processes
- ✅ **Modular**: Clean separation between solver, communication, and data structures
- ✅ **Practical**: Optimized for time-stepping PDE solvers
- ✅ **Extensible**: Easy to integrate into larger simulation codes
- ✅ **Tested**: Comprehensive benchmarking and validation tools

The design choices balance performance, maintainability, and usability, making it suitable for both educational purposes and real-world scientific computing applications.

---

## References

1. **Schur Complement Method**: Lecture 5, Slides 26-32
2. **Thomas Algorithm**: Conte & de Boor, "Elementary Numerical Analysis"
3. **MPI Programming**: Gropp, Lusk, Skjellum, "Using MPI"
4. **Domain Decomposition**: Toselli & Widlund, "Domain Decomposition Methods"

---

**Last Updated**: 2026-01-08
**Version**: 1.0
**Author**: HPSCA Project Module A
