# Exhaustive Bug Analysis: Schur Complement Parallel Solver

## Executive Summary

The parallel Schur complement solver works correctly for **single process** (np=1) but produces **completely wrong results** for multiple processes (np≥2). The root cause is an **off-by-one error** in the domain decomposition logic in [SchurMatrixData.hpp:127-171](inc/SchurMatrixData.hpp#L127-L171).

**Error Magnitude:**
- np=1: Error = 0 (exact match with serial)
- np=2: Max error ≈ 1029, L2 error ≈ 5954

---

## The Problem

### What Should Happen

In domain decomposition with interface overlap, adjacent processes should **share interface points**:

```
Process 0: [====|]     Process 1: [|====]
           0    49            49   99
                ↑              ↑
                Same point (interface)
```

- Rank 0 owns global points [0..49], with right interface at index 49
- Rank 1 owns global points [49..99], with left interface at index 49
- **Point 49 is shared** between both processes

### What Actually Happens

```
Process 0: [=====|]    Process 1: [|====]
           0     50           49   99
                 ↑             ↑
           global[50]    global[49]
           DIFFERENT POINTS!
```

- Rank 0 owns global points [0..50], right interface at 50
- Rank 1 owns global points [49..99], left interface at 49
- **Interfaces don't overlap** - they're at different global indices!

---

## Root Cause Analysis

### The Buggy Code

Location: [inc/SchurMatrixData.hpp:132-143](inc/SchurMatrixData.hpp#L132-L143)

```cpp
// Compute local size with interface overlap
// Each process owns (global_N - 1) / num_procs + 1 points
// Plus 1 overlap point at each interface (shared with neighbor)
Dim base_size = (global_N - 1) / num_procs;
Dim remainder = (global_N - 1) % num_procs;

// Distribute remainder to first 'remainder' processes
if (rank < static_cast<int>(remainder)) {
    local_N = base_size + 2;  // ← BUG HERE!
} else {
    local_N = base_size + 1;  // ← AND HERE!
}
```

### Why It's Wrong

For **N=100, np=2**:

```
base_size = (100-1) / 2 = 49
remainder = (100-1) % 2 = 1

Rank 0 (rank < remainder):
  local_N = 49 + 2 = 51 points
  Owns global indices [0..50]
  Right interface = global[50]

Rank 1 (rank >= remainder):
  local_N = 49 + 1 = 50 points
  global_start = 50  (computed as sum of internal points)
  Owns global indices [49..98]  (local[0..49] → global[49..98])
  Left interface = global[49]
```

**Result:** Rank 0's right interface (50) ≠ Rank 1's left interface (49)

### The Conceptual Error

The formula tries to distribute **(N-1)** points across processes, then add overlap. But it incorrectly assumes:
1. `base_size` is the number of **internal** points per process
2. Adding "+1" or "+2" accounts for interface overlap

This is **wrong** because:
- The "+2" for `rank < remainder` means "extra internal point + right interface"
- But this creates a gap between rank 0's right interface and rank 1's left interface
- The formula doesn't properly account for the fact that **interfaces are shared points**

---

## Detailed Trace for N=100, np=2

### Step 1: Domain Decomposition

```
Rank 0:
├─ local_N = 51
├─ global_start = 0
├─ n_internal = 50  (local_N - 1, because has_right_interface=true)
├─ Storage: local[0..50]
└─ Mapping:
    local[0..49]  → global[0..49]   (internal points)
    local[50]     → global[50]      (RIGHT INTERFACE)

Rank 1:
├─ local_N = 50
├─ global_start = 50  (sum of rank 0's internal: 50)
├─ n_internal = 49  (local_N - 1, because has_left_interface=true)
├─ Storage: local[0..49]
└─ Mapping:
    local[0]      → global[49]      (LEFT INTERFACE)  ← WRONG!
    local[1..49]  → global[50..98]  (internal points)
```

**Problem:** Rank 1 expects its left interface at global[49], but rank 0's right interface is at global[50].

### Step 2: Coefficient Distribution

In [run_parallel_solver.cpp:503-521](tests/run_parallel_solver.cpp#L503-L521):

```cpp
for (int i = 0; i < local_N; ++i) {
    int global_i;
    if (rank == 0) {
        global_i = i;  // Rank 0: local[i] → global[i]
    } else {
        global_i = global_start - 1 + i;  // Rank 1: local[i] → global[49+i]
    }

    a_local[i] = a_global[global_i];
    b_local[i] = b_global[global_i];
    c_local[i] = c_global[global_i];
    rhs_local[i] = rhs_global[global_i];
}
```

**Rank 0 gets:**
- local[0] = global[0]: a=-0, b=2, c=-1, rhs=sin(0)
- local[1] = global[1]: a=-1, b=2, c=-1, rhs=sin(π/100)
- ...
- local[50] = global[50]: a=-1, b=2, c=-1, rhs=sin(50π/100)

**Rank 1 gets:**
- local[0] = global[49]: a=-1, b=2, c=-1, rhs=sin(49π/100)  ← Should be interface!
- local[1] = global[50]: a=-1, b=2, c=-1, rhs=sin(50π/100)
- ...
- local[49] = global[98]: a=-1, b=2, c=-1, rhs=sin(98π/100)

**Result:** Point global[50] appears in **both** processes (rank 0's local[50] and rank 1's local[1]), but global[50] is NOT an interface point! The actual interface should be at global[49] or global[50], but consistently.

### Step 3: Schur Complement Construction

The Schur complement method assumes:
1. Each process has **internal points** (not shared)
2. Processes share **interface points** at boundaries
3. The reduced system is assembled using these shared interfaces

With the misaligned interfaces:
- Rank 0 thinks the interface is at global[50]
- Rank 1 thinks the interface is at global[49]
- The reduced system mixes contributions from **two different points**
- The solution is completely corrupted

### Step 4: Solution Assembly

When gathering the solution in [run_parallel_solver.cpp:102-135](tests/run_parallel_solver.cpp#L102-L135):

```cpp
// Rank 0 stores global[0..50]
// Rank 1 sends local[1..49] (skipping local[0] to avoid duplication)
//   → This sends global[50..98]

Final assembly:
  global[0..50]   from rank 0  (51 points)
  global[50..98]  from rank 1  (49 points)
```

**Problem:** Global[50] appears in both! And global[99] is missing!

---

## Impact on Solution Quality

### Test Case

Tridiagonal system:
```
-u[i-1] + 2*u[i] - u[i+1] = sin(i*π/N)
```

With boundary conditions: `a[0] = 0`, `c[N-1] = 0`

### Results

**Serial (correct):**
```
x[0]  = 31.513212
x[1]  = 63.026424
x[2]  = 94.508224
x[49] = 645.55...
x[50] = 661.82...
```

**Parallel np=2 (wrong):**
```
x[0]  = 11.335748   (should be 31.513)  Error: -20.18
x[1]  = 22.671495   (should be 63.026)  Error: -40.35
x[2]  = 33.975830   (should be 94.508)  Error: -60.53
x[49] = 532.91...   (should be 645.55)  Error: -112.6
x[50] = 545.29...   (should be 661.82)  Error: -116.5
```

**Error statistics:**
- Maximum error: 1029.05
- L2 norm: 5954.1
- Relative error: ~36%

The solution is **completely wrong** - not just slightly inaccurate, but fundamentally broken.

---

## Why np=1 Works

With **single process** (np=1):
- No interfaces (num_interfaces = 0)
- No domain decomposition
- The Schur solver degenerates to standard Thomas algorithm
- Works correctly

This is why the test passes for np=1 but fails for np≥2.

---

## The Correct Formula

The domain decomposition should work as follows:

### Conceptual Approach

1. **Partition internal points:** Divide **(N-2)** internal points (excluding global boundaries)
2. **Add boundaries:** First process gets left boundary, last process gets right boundary
3. **Add interfaces:** Non-boundary processes get overlap points

### Example for N=100, np=2

```
Total points: 100 [0..99]
Boundaries: global[0] and global[99]
Internal + interfaces: 98 points [1..98]

Desired partition:
  Rank 0: [0..49]  (50 points: boundary + 49 internal/interface)
  Rank 1: [49..99] (51 points: interface + 49 internal + boundary)

  Interface at global[49] is shared
```

### Corrected Formula

```cpp
// For 2 processes, simple split:
if (num_procs == 2) {
    Dim split_point = global_N / 2;

    if (rank == 0) {
        local_N = split_point + 1;  // [0..split_point]
        global_start = 0;
    } else {
        local_N = global_N - split_point;  // [split_point..N-1]
        global_start = split_point;
    }
}

// For general case, need to carefully handle overlaps
```

Or alternatively, use a **non-overlapping partition** and handle interface communication explicitly.

---

## Files Affected

1. **[inc/SchurMatrixData.hpp](inc/SchurMatrixData.hpp#L132-143)**
   - Lines 132-143: Domain decomposition logic (the bug)
   - Lines 166-171: `global_start` computation (depends on buggy local_N)

2. **[tests/run_parallel_solver.cpp](tests/run_parallel_solver.cpp)**
   - Lines 69-84: Local coefficient extraction (tries to work around the bug)
   - Lines 102-135: Solution gathering (wrong because of misaligned interfaces)

3. **[inc/SchurComplementSolver.hpp](inc/SchurComplementSolver.hpp)**
   - Assumes correct domain decomposition from SchurMatrixData
   - Works correctly IF given proper input

---

## Verification

### Test 1: Single Process
```bash
mpirun -np 1 ./bin/run_parallel_solver 100
awk -f compare.awk solution_serial_N100.dat solution_parallel_N100_np1.dat
```
**Result:** Error = 0 (exact match)

### Test 2: Two Processes
```bash
mpirun -np 2 ./bin/run_parallel_solver 100
awk -f compare.awk solution_serial_N100.dat solution_parallel_N100_np2.dat
```
**Result:** Max error ≈ 1029 (completely wrong)

---

## Recommended Fix

### Option 1: Fix the Overlap Formula

Redesign `SchurMatrixData::init()` to properly handle overlapping interfaces:

```cpp
void init(Dim global_size, int nprocs, int rank) {
    global_N = global_size;
    num_procs = nprocs;
    my_rank = rank;

    if (nprocs == 1) {
        // Single process - no decomposition
        local_N = global_N;
        global_start = 0;
        n_internal = global_N;
        has_left_interface = false;
        has_right_interface = false;
    } else {
        // Multi-process with proper interface sharing
        Dim points_per_proc = global_N / nprocs;
        Dim remainder = global_N % nprocs;

        // Each rank gets base points, first 'remainder' ranks get +1
        Dim base_local_N = points_per_proc + (rank < remainder ? 1 : 0);

        // Compute global starting index (non-overlapping)
        global_start = rank * points_per_proc + std::min(rank, (int)remainder);

        // Add overlap for interior processes
        if (rank > 0) {
            base_local_N++;  // Add left interface overlap
            global_start--;  // Shift to include left interface
        }
        if (rank < nprocs - 1) {
            base_local_N++;  // Add right interface overlap
        }

        local_N = base_local_N;
        // ... set interface flags and n_internal
    }
}
```

### Option 2: Use Non-Overlapping Partitions

Simpler approach - partition without overlap and handle interface communication explicitly:

```cpp
// Non-overlapping partition
Dim points_per_proc = global_N / nprocs;
Dim remainder = global_N % nprocs;

local_N = points_per_proc + (rank < remainder ? 1 : 0);
global_start = rank * points_per_proc + std::min(rank, (int)remainder);

// Communicate with neighbors to get interface values
// No duplication in storage
```

---

## Summary

**The Bug:**
- Off-by-one error in domain decomposition formula
- Interfaces don't align between adjacent processes
- Rank i's right interface ≠ Rank i+1's left interface

**Manifestation:**
- Works perfectly for np=1 (no decomposition needed)
- Completely breaks for np≥2 (misaligned interfaces corrupt the solution)

**Root Cause:**
- Incorrect formula: `local_N = base_size + 2` for ranks with remainder
- Should be: proper accounting for shared interface points

**Impact:**
- Solution error ~1000 for N=100, np=2
- Not a precision issue - fundamentally wrong decomposition

**Fix Required:**
- Redesign domain decomposition in `SchurMatrixData::init()`
- Ensure adjacent processes share the same global index for interfaces
- Update coefficient distribution and solution gathering accordingly
