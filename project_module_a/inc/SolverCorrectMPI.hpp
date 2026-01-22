#ifndef SOLVER_HPP
#define SOLVER_HPP
#include <string>
#include <cmath>
#include <functional> // For std::function/lambdas
#ifdef _OPENMP
#include <omp.h>
#endif
#include "Variables.hpp"
#include "ScalarVariable.hpp"
#include "VectorVariable.hpp"
#include "DimensionHandler.hpp"
#include "BoundaryFunctions.hpp"
#include "MPITopology3D.hpp"
#include "SchurComplementSolver.hpp"

constexpr bool DEBUG_BLOCK = false; // set to true to enable debug prints

class Solver
{
public:
    // Pure virtual destructor makes the class abstract
    virtual ~Solver() = 0;
    Solver(Dim Nx_, Dim Ny_, Dim Nz_, Real dx_, Real dy_, Real dz_, Real dt_)
        : Nx(Nx_), Ny(Ny_), Nz(Nz_),
          dx(dx_), dy(dy_), dz(dz_), dt(dt_) { t = dt_; }; // solve doesn't solve for t=0 called firstly at t=dt

    template <Dim direction>
    void solve(ScalarVariable &rhs, ScalarVariable &solution, const DimensionsHandlerScalar &dim_handler);
    void advance_time()
    {
        t += dt;
    }

    void set_t(Real t)
    {
        this->t = t;
    }

protected:
    Dim Nx; // Grid points in x
    Dim Ny; // Grid points in y
    Dim Nz; // Grid points in z

    Real dx; // Grid spacing in x
    Real dy; // Grid spacing in y
    Real dz; // Grid spacing in z

    Real t;
    Real dt;
    // Make available to derived classes
    void thomas_algorithm(const std::vector<Real> &a, const std::vector<Real> &b, const std::vector<Real> &c, const std::vector<Real> &rhs, std::vector<Real> &x)
    {
        int n = rhs.size();
        std::vector<Real> c_prime(c.size(), 0.0);
        std::vector<Real> rhs_prime(n, 0.0);

        c_prime[0] = c[0] / b[0];
        rhs_prime[0] = rhs[0] / b[0];

        for (int i = 1; i < n; ++i)
        {
            Real m = Real(1.0) / (b[i] - a[i] * c_prime[i - 1]);
            c_prime[i] = c[i] * m;
            rhs_prime[i] = (rhs[i] - a[i] * rhs_prime[i - 1]) * m;
        }

        x[n - 1] = rhs_prime[n - 1];

        for (int i = n - 2; i >= 0; --i)
        {
            x[i] = rhs_prime[i] - c_prime[i] * x[i + 1];
        }
    }
};

// Definition of pure virtual destructor
inline Solver::~Solver() {}

// =============================================================================================
// ====================================Pressure Solver Class====================================
// =============================================================================================

class PressureSolver : public Solver
{
public:
    BoundaryFunctions &p_boundary;

    PressureSolver(Dim Nx_, Dim Ny_, Dim Nz_,
                   Real dx_, Real dy_, Real dz_,
                   Real dt_,
                   BoundaryFunctions &p_boundary_)
        : Solver(Nx_, Ny_, Nz_, dx_, dy_, dz_, dt_), p_boundary(p_boundary_) {}

    BoundaryFunctions &set_p_boundary() { return p_boundary; }
    // =============================================================
    // PRESSURE SWEEP (scalar) - FAST + BATCHED + OpenMP
    // - builds RHS only for local_N
    // - batched Schur solve (one MPI collective per batch, not per line)
    // - OpenMP used in pack + unpack loops
    // =============================================================

    void solve_x_mpi(ScalarVariable &rhs,
                     ScalarVariable &solution,
                     const MPITopology3D &topo,
                     bool use_omp = false)
    {
        const Dim N = Nx;
        const Real h = dx;
        const Real alpha = Real(1.0) / (h * h);

        // ---------------------------
        // Communicator in X
        // ---------------------------
        MPI_Comm comm_x_raw = topo.comm_x();
        int rank_x = 0, size_x = 1;
        MPI_Comm_rank(comm_x_raw, &rank_x);
        MPI_Comm_size(comm_x_raw, &size_x);
        MPICommunicator comm_x(comm_x_raw, false);

        // ---------------------------
        // Build global a,b,c once (scalar)
        // (I - dxx) with Neumann via ghost elimination
        // ---------------------------
        std::vector<Real> a_full(N, -alpha);
        std::vector<Real> b_full(N, Real(1.0) + Real(2.0) * alpha);
        std::vector<Real> c_full(N, -alpha);

        // left Neumann: (1+2α) ψ0 - 2α ψ1 = rhs0 - 2 gL/h
        a_full[0] = Real(0.0);
        c_full[0] = -Real(2.0) * alpha;

        // right Neumann: -2α ψ_{N-2} + (1+2α) ψ_{N-1} = rhs_{N-1} + 2 gR/h
        a_full[N - 1] = -Real(2.0) * alpha;
        c_full[N - 1] = Real(0.0);

        // ---------------------------
        // Schur solver preprocess (once)
        // ---------------------------
        SchurComplementSolver schur(N, size_x, rank_x, comm_x);

        const int local_N = schur.get_local_N();
        const int global_start = schur.get_global_start();

        std::vector<Real> a_local(local_N, 0.0), b_local(local_N, 0.0), c_local(local_N, 0.0);
        for (int il = 0; il < local_N; ++il)
        {
            const int ig = global_start + il;
            if (ig < 0 || ig >= (int)N)
                continue;
            a_local[il] = a_full[ig];
            b_local[il] = b_full[ig];
            c_local[il] = c_full[ig];
        }
        schur.preprocess(a_local, b_local, c_local);

        // ---------------------------
        // Local (j,k) slab for this rank (in Py,Pz)
        // ---------------------------
        Dim j0 = topo.local_j0(Ny), j1 = topo.local_j1(Ny);
        Dim k0 = topo.local_k0(Nz), k1 = topo.local_k1(Nz);

        const int nJ = int(j1 - j0);
        const int nK = int(k1 - k0);
        const int nLinesLocal = nJ * nK;

        auto lid_of = [&](Dim j, Dim k) -> int
        { return int((k - k0) * nJ + (j - j0)); };

        auto jk_of_lid = [&](int lid, Dim &j, Dim &k)
        {
            k = k0 + Dim(lid / nJ);
            j = j0 + Dim(lid - int(k - k0) * nJ);
        };

        // write-back: avoid duplicate left interface
        const int i0_local = (rank_x == 0) ? 0 : 1;

        // ------------------------------------------------------------
        // Build RHS only for local_N entries (FAST)
        // Neumann contributions:
        //   d[0]     -= 2*gL/h
        //   d[N-1]   += 2*gR/h
        // ------------------------------------------------------------
        auto build_rhs_local = [&](Dim j, Dim k, Real *out)
        {
            const Real y = Real(j) * dy;
            const Real z = Real(k) * dz;

            // g = dψ/dx on boundaries
            const Real gL = p_boundary.value<0>(Real(0.0), y, z, t);
            const Real gR = p_boundary.value<0>((Nx - Real(0.5)) * dx, y, z, t);

            for (int il = 0; il < local_N; ++il)
            {
                const int ig = global_start + il;

                if (ig < 0 || ig >= (int)N)
                {
                    out[il] = Real(0.0);
                    continue;
                }

                Real val = rhs.get(Dim(ig), j, k);

                if (ig == 0)
                    val -= Real(2.0) * gL / h;
                else if (ig == N - 1)
                    val += Real(2.0) * gR / h;

                out[il] = val;
            }
        };

        // ------------------------------------------------------------
        // Batched solve over all (j,k) lines
        // ------------------------------------------------------------
        std::vector<int> active_lids;
        active_lids.reserve(nLinesLocal);

        for (Dim k = k0; k < k1; ++k)
            for (Dim j = j0; j < j1; ++j)
                active_lids.push_back(lid_of(j, k));

        const int nActive = int(active_lids.size());
        if (nActive == 0)
            return;

        std::vector<Real> rhs_flat(size_t(nActive) * size_t(local_N));
        std::vector<Real> sol_flat(size_t(nActive) * size_t(local_N));

#pragma omp parallel for schedule(static) if (use_omp)
        for (int p = 0; p < nActive; ++p)
        {
            Dim j, k;
            jk_of_lid(active_lids[p], j, k);
            build_rhs_local(j, k, rhs_flat.data() + size_t(p) * size_t(local_N));
        }

        schur.solve_batch(rhs_flat.data(), nActive, sol_flat.data(), use_omp);

#pragma omp parallel for schedule(static) if (use_omp)
        for (int p = 0; p < nActive; ++p)
        {
            Dim j, k;
            jk_of_lid(active_lids[p], j, k);

            const Real *x = sol_flat.data() + size_t(p) * size_t(local_N);
            for (int il = i0_local; il < local_N; ++il)
            {
                const int ig = global_start + il;
                if (ig < 0 || ig >= (int)N)
                    continue;
                solution.set(Dim(ig), j, k) = x[il];
            }
        }
    }

    void solve_y_mpi(ScalarVariable &rhs,
                     ScalarVariable &solution,
                     const MPITopology3D &topo,
                     bool use_omp = false)
    {
        const Dim N = Ny;
        const Real h = dy;
        const Real alpha = Real(1.0) / (h * h);

        // communicator in Y
        MPI_Comm comm_y_raw = topo.comm_y();
        int rank_y = 0, size_y = 1;
        MPI_Comm_rank(comm_y_raw, &rank_y);
        MPI_Comm_size(comm_y_raw, &size_y);
        MPICommunicator comm_y(comm_y_raw, false);

        // a,b,c with Neumann via ghost elimination
        std::vector<Real> a_full(N, -alpha);
        std::vector<Real> b_full(N, Real(1.0) + Real(2.0) * alpha);
        std::vector<Real> c_full(N, -alpha);

        a_full[0] = Real(0.0);
        c_full[0] = -Real(2.0) * alpha;
        a_full[N - 1] = -Real(2.0) * alpha;
        c_full[N - 1] = Real(0.0);

        SchurComplementSolver schur(N, size_y, rank_y, comm_y);
        const int local_N = schur.get_local_N();
        const int global_start = schur.get_global_start();

        std::vector<Real> a_local(local_N, 0.0), b_local(local_N, 0.0), c_local(local_N, 0.0);
        for (int il = 0; il < local_N; ++il)
        {
            const int ig = global_start + il;
            if (ig < 0 || ig >= (int)N)
                continue;
            a_local[il] = a_full[ig];
            b_local[il] = b_full[ig];
            c_local[il] = c_full[ig];
        }
        schur.preprocess(a_local, b_local, c_local);

        // local (i,k) slab
        Dim i0 = topo.local_i0(Nx), i1 = topo.local_i1(Nx);
        Dim k0 = topo.local_k0(Nz), k1 = topo.local_k1(Nz);

        const int nI = int(i1 - i0);
        const int nK = int(k1 - k0);
        const int nLinesLocal = nI * nK;

        auto lid_of = [&](Dim i, Dim k) -> int
        { return int((k - k0) * nI + (i - i0)); };

        auto ik_of_lid = [&](int lid, Dim &i, Dim &k)
        {
            k = k0 + Dim(lid / nI);
            i = i0 + Dim(lid - int(k - k0) * nI);
        };

        const int j0_local = (rank_y == 0) ? 0 : 1;

        auto build_rhs_local = [&](Dim i, Dim k, Real *out)
        {
            const Real x = Real(i) * dx;
            const Real z = Real(k) * dz;

            const Real gB = p_boundary.value<1>(x, Real(0.0), z, t);
            const Real gT = p_boundary.value<1>(x, (Ny - Real(0.5)) * dy, z, t);

            for (int il = 0; il < local_N; ++il)
            {
                const int ig = global_start + il;
                if (ig < 0 || ig >= (int)N)
                {
                    out[il] = Real(0.0);
                    continue;
                }

                Real val = rhs.get(i, Dim(ig), k);

                if (ig == 0)
                    val -= Real(2.0) * gB / h;
                else if (ig == N - 1)
                    val += Real(2.0) * gT / h;

                out[il] = val;
            }
        };

        std::vector<int> active_lids;
        active_lids.reserve(nLinesLocal);
        for (Dim k = k0; k < k1; ++k)
            for (Dim i = i0; i < i1; ++i)
                active_lids.push_back(lid_of(i, k));

        const int nActive = int(active_lids.size());
        if (nActive == 0)
            return;

        std::vector<Real> rhs_flat(size_t(nActive) * size_t(local_N));
        std::vector<Real> sol_flat(size_t(nActive) * size_t(local_N));

#pragma omp parallel for schedule(static) if (use_omp)
        for (int p = 0; p < nActive; ++p)
        {
            Dim i, k;
            ik_of_lid(active_lids[p], i, k);
            build_rhs_local(i, k, rhs_flat.data() + size_t(p) * size_t(local_N));
        }

        schur.solve_batch(rhs_flat.data(), nActive, sol_flat.data(), use_omp);

#pragma omp parallel for schedule(static) if (use_omp)
        for (int p = 0; p < nActive; ++p)
        {
            Dim i, k;
            ik_of_lid(active_lids[p], i, k);

            const Real *x = sol_flat.data() + size_t(p) * size_t(local_N);
            for (int il = j0_local; il < local_N; ++il)
            {
                const int ig = global_start + il;
                if (ig < 0 || ig >= (int)N)
                    continue;
                solution.set(i, Dim(ig), k) = x[il];
            }
        }
    }

    void solve_z_mpi(ScalarVariable &rhs,
                     ScalarVariable &solution,
                     const MPITopology3D &topo,
                     bool use_omp = false)
    {
        const Dim N = Nz;
        const Real h = dz;
        const Real alpha = Real(1.0) / (h * h);

        // communicator in Z
        MPI_Comm comm_z_raw = topo.comm_z();
        int rank_z = 0, size_z = 1;
        MPI_Comm_rank(comm_z_raw, &rank_z);
        MPI_Comm_size(comm_z_raw, &size_z);
        MPICommunicator comm_z(comm_z_raw, false);

        // a,b,c with Neumann via ghost elimination
        std::vector<Real> a_full(N, -alpha);
        std::vector<Real> b_full(N, Real(1.0) + Real(2.0) * alpha);
        std::vector<Real> c_full(N, -alpha);

        a_full[0] = Real(0.0);
        c_full[0] = -Real(2.0) * alpha;
        a_full[N - 1] = -Real(2.0) * alpha;
        c_full[N - 1] = Real(0.0);

        SchurComplementSolver schur(N, size_z, rank_z, comm_z);
        const int local_N = schur.get_local_N();
        const int global_start = schur.get_global_start();

        std::vector<Real> a_local(local_N, 0.0), b_local(local_N, 0.0), c_local(local_N, 0.0);
        for (int il = 0; il < local_N; ++il)
        {
            const int ig = global_start + il;
            if (ig < 0 || ig >= (int)N)
                continue;
            a_local[il] = a_full[ig];
            b_local[il] = b_full[ig];
            c_local[il] = c_full[ig];
        }
        schur.preprocess(a_local, b_local, c_local);

        // local (i,j) slab
        Dim i0 = topo.local_i0(Nx), i1 = topo.local_i1(Nx);
        Dim j0 = topo.local_j0(Ny), j1 = topo.local_j1(Ny);

        const int nI = int(i1 - i0);
        const int nJ = int(j1 - j0);
        const int nLinesLocal = nI * nJ;

        auto lid_of = [&](Dim i, Dim j) -> int
        { return int((j - j0) * nI + (i - i0)); };

        auto ij_of_lid = [&](int lid, Dim &i, Dim &j)
        {
            j = j0 + Dim(lid / nI);
            i = i0 + Dim(lid - int(j - j0) * nI);
        };

        const int k0_local = (rank_z == 0) ? 0 : 1;

        auto build_rhs_local = [&](Dim i, Dim j, Real *out)
        {
            const Real x = Real(i) * dx;
            const Real y = Real(j) * dy;

            const Real gF = p_boundary.value<2>(x, y, Real(0.0), t);
            const Real gB = p_boundary.value<2>(x, y, (Nz - Real(0.5)) * dz, t);

            for (int il = 0; il < local_N; ++il)
            {
                const int ig = global_start + il;
                if (ig < 0 || ig >= (int)N)
                {
                    out[il] = Real(0.0);
                    continue;
                }

                Real val = rhs.get(i, j, Dim(ig));

                if (ig == 0)
                    val -= Real(2.0) * gF / h;
                else if (ig == N - 1)
                    val += Real(2.0) * gB / h;

                out[il] = val;
            }
        };

        std::vector<int> active_lids;
        active_lids.reserve(nLinesLocal);
        for (Dim j = j0; j < j1; ++j)
            for (Dim i = i0; i < i1; ++i)
                active_lids.push_back(lid_of(i, j));

        const int nActive = int(active_lids.size());
        if (nActive == 0)
            return;

        std::vector<Real> rhs_flat(size_t(nActive) * size_t(local_N));
        std::vector<Real> sol_flat(size_t(nActive) * size_t(local_N));

#pragma omp parallel for schedule(static) if (use_omp)
        for (int p = 0; p < nActive; ++p)
        {
            Dim i, j;
            ij_of_lid(active_lids[p], i, j);
            build_rhs_local(i, j, rhs_flat.data() + size_t(p) * size_t(local_N));
        }

        schur.solve_batch(rhs_flat.data(), nActive, sol_flat.data(), use_omp);

#pragma omp parallel for schedule(static) if (use_omp)
        for (int p = 0; p < nActive; ++p)
        {
            Dim i, j;
            ij_of_lid(active_lids[p], i, j);

            const Real *x = sol_flat.data() + size_t(p) * size_t(local_N);
            for (int il = k0_local; il < local_N; ++il)
            {
                const int ig = global_start + il;
                if (ig < 0 || ig >= (int)N)
                    continue;
                solution.set(i, j, Dim(ig)) = x[il];
            }
        }
    }
};

// =============================================================================================
// ====================================Velocity Solver Class====================================
// =============================================================================================

class VelocitySolver : public Solver
{
public:
    ScalarVariable &gamma_field;
    BoundaryFunctions &u_boundary;
    VelocitySolver(Dim Nx_, Dim Ny_, Dim Nz_, Real dx_, Real dy_, Real dz_, Real dt_, ScalarVariable &gam, BoundaryFunctions &u_bnd)
        : Solver(Nx_, Ny_, Nz_, dx_, dy_, dz_, dt_), gamma_field(gam), u_boundary(u_bnd) {}

    void set_gamma(ScalarVariable &g) { gamma_field = g; }
    BoundaryFunctions &set_u_boundary() { return u_boundary; }
    template <Dim direction>
    bool is_known_face(Dim index_1, Dim index_2, Dim component) const
    {
        if constexpr (direction == 0)
        {
            if (component == 0)
                return (index_1 == 0 || index_2 == 0);
            if (component == 1)
                return (index_1 == Ny - 1 || index_2 == 0);
            if (component == 2)
                return (index_1 == 0 || index_2 == Nz - 1);
        }
        else if constexpr (direction == 1) // Sweep Y-direction
        {
            if (component == 0)
                return (index_1 == Nx - 1 || index_2 == 0);
            if (component == 1)
                return (index_1 == 0 || index_2 == 0);
            if (component == 2)
                return (index_1 == 0 || index_2 == Nz - 1);
        }
        else
        { // direction == 2
            if (component == 0)
                return (index_1 == Nx - 1 || index_2 == 0);
            if (component == 1)
                return (index_1 == 0 || index_2 == Ny - 1);
            if (component == 2)
                return (index_1 == 0 || index_2 == 0);
        }
        return false;
    }
    template <Dim direction>
    bool handle_known_face(VectorVariable &solution, Dim index_1, Dim index_2, Dim component)
    {
        if (!is_known_face<direction>(index_1, index_2, component))
            return false;

        auto update_bc = [&](Dim i, Dim j, Dim k)
        {
            Real x = i * dx, y = j * dy, z = k * dz;

            // BoundaryFunctions::value is now thread-safe via thread_local parser
            if (component == 0)
                solution.set(0, i, j, k) = u_boundary.value<0>(x + dx / Real(2.0), y, z, t);
            else if (component == 1)
                solution.set(1, i, j, k) = u_boundary.value<1>(x, y + dy / Real(2.0), z, t);
            else if (component == 2)
                solution.set(2, i, j, k) = u_boundary.value<2>(x, y, z + dz / Real(2.0), t);
        };

        if constexpr (direction == 0)
        {
            for (Dim i = 0; i < Nx; ++i)
            {
                update_bc(i, index_1, index_2);
            }
        }
        else if constexpr (direction == 1)
        {
            for (Dim j = 0; j < Ny; ++j)
                update_bc(index_1, j, index_2);
        }
        else
        {
            for (Dim k = 0; k < Nz; ++k)
                update_bc(index_1, index_2, k);
        }

        return true;
    }

    template <Dim direction>
    void apply_bc(VectorVariable &rhs)
    {
        // Domain lengths:
        Real Lx = dx * (Nx - Real(0.5));
        Real Ly = dy * (Ny - Real(0.5));
        Real Lz = dz * (Nz - Real(0.5));

        if constexpr (direction == 0)
        {
            for (Dim index_1 = 0; index_1 < Ny; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Nz; ++index_2)
                {
                    // on comp1 we have normal components
                    rhs.set(direction, 0, index_1, index_2) = (u_boundary.value<direction>(0, index_1 * dy, index_2 * dz, t)) - ((u_boundary.first_derivative<1>(0, index_1 * dy, index_2 * dz, t, dy)) + (u_boundary.first_derivative<2>(0, index_1 * dy, index_2 * dz, t, dz))) * dx * Real(0.5);
                    rhs.set(direction, Nx - 1, index_1, index_2) = u_boundary.value<direction>(Lx, index_1 * dy, index_2 * dz, t);

                    // on comp2 we have tangent components
                    rhs.set(1, 0, index_1, index_2) = u_boundary.value<1>(0, 0.5 * dy + index_1 * dy, index_2 * dz, t);
                    rhs.set(1, Nx - 1, index_1, index_2) = rhs.value(1, Nx - 1, index_1, index_2) + Real(2.0) * gamma_field.get(Nx - 1, index_1, index_2) / (dx * dx) * (u_boundary.value<1>(Lx, 0.5 * dy + index_1 * dy, index_2 * dz, t));

                    // on comp3 we have tangent components
                    rhs.set(2, 0, index_1, index_2) = u_boundary.value<2>(0, index_1 * dy, 0.5 * dz + index_2 * dz, t);
                    rhs.set(2, Nx - 1, index_1, index_2) = rhs.value(2, Nx - 1, index_1, index_2) + Real(2.0) * gamma_field.get(Nx - 1, index_1, index_2) / (dx * dx) * (u_boundary.value<2>(Lx, index_1 * dy, 0.5 * dz + index_2 * dz, t));
                }
            }
        }

        else if constexpr (direction == 1)
        {
            for (Dim index_1 = 0; index_1 < Nx; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Nz; ++index_2)
                {
                    // on comp2 we have normal components
                    rhs.set(direction, index_1, 0, index_2) = (u_boundary.value<direction>(index_1 * dx, 0, index_2 * dz, t)) - ((u_boundary.first_derivative<0>(index_1 * dx, 0, index_2 * dz, t, dx)) + (u_boundary.first_derivative<2>(index_1 * dx, 0, index_2 * dz, t, dz))) * dy * Real(0.5);
                    rhs.set(direction, index_1, Ny - 1, index_2) = u_boundary.value<direction>(index_1 * dx, Ly, index_2 * dz, t);

                    // on comp1 we have tangent components
                    rhs.set(0, index_1, 0, index_2) = u_boundary.value<0>(0.5 * dx + index_1 * dx, 0, index_2 * dz, t);
                    rhs.set(0, index_1, Ny - 1, index_2) = rhs.value(0, index_1, Ny - 1, index_2) + Real(2.0) * gamma_field.get(index_1, Ny - 1, index_2) / (dy * dy) * (u_boundary.value<0>(0.5 * dx + index_1 * dx, Ly, index_2 * dz, t));
                    // on comp3 we have tangent components
                    rhs.set(2, index_1, 0, index_2) = u_boundary.value<2>(index_1 * dx, 0, 0.5 * dz + index_2 * dz, t);
                    rhs.set(2, index_1, Ny - 1, index_2) = rhs.value(2, index_1, Ny - 1, index_2) + Real(2.0) * gamma_field.get(index_1, Ny - 1, index_2) / (dy * dy) * (u_boundary.value<2>(index_1 * dx, Ly, 0.5 * dz + index_2 * dz, t));
                }
            }
        }
        else if constexpr (direction == 2)
        {
            for (Dim index_1 = 0; index_1 < Nx; ++index_1)
            {
                for (Dim index_2 = 0; index_2 < Ny; ++index_2)
                {
                    // on comp3 we have normal components
                    rhs.set(direction, index_1, index_2, 0) = (u_boundary.value<direction>(index_1 * dx, index_2 * dy, 0, t)) - (((u_boundary.first_derivative<0>(index_1 * dx, index_2 * dy, 0, t, dx))) + (u_boundary.first_derivative<1>(index_1 * dx, index_2 * dy, 0, t, dy))) * dz * Real(0.5);
                    rhs.set(direction, index_1, index_2, Nz - 1) = u_boundary.value<direction>(index_1 * dx, index_2 * dy, Lz, t);

                    // on comp1 we have tangent components
                    rhs.set(0, index_1, index_2, 0) = (u_boundary.value<0>(0.5 * dx + index_1 * dx, index_2 * dy, 0, t));
                    rhs.set(0, index_1, index_2, Nz - 1) = rhs.value(0, index_1, index_2, Nz - 1) + Real(2.0) * gamma_field.get(index_1, index_2, Nz - 1) / (dz * dz) * (u_boundary.value<0>(0.5 * dx + index_1 * dx, index_2 * dy, Lz, t));

                    // on comp2 we have tangent components
                    rhs.set(1, index_1, index_2, 0) = (u_boundary.value<1>(index_1 * dx, 0.5 * dy + index_2 * dy, 0, t));
                    rhs.set(1, index_1, index_2, Nz - 1) = rhs.value(1, index_1, index_2, Nz - 1) + Real(2.0) * gamma_field.get(index_1, index_2, Nz - 1) / (dz * dz) * (u_boundary.value<1>(index_1 * dx, 0.5 * dy + index_2 * dy, Lz, t));
                }
            }
        }
    };

    // =============================================================
    // X-DIRECTION ONLY SOLVE (direction splitting validation)
    // Solves: (I - gamma * dxx) u = rhs   along x
    //
    // FAST VERSION:
    // - does NOT call apply_bc<0>(rhs)
    // - instead builds RHS *locally* per line (only local_N entries)
    // - boundary closure matches apply_bc<0> (analytic derivatives)
    // - known lines handled exactly like the correct solver
    // =============================================================
    void solve_x_only(VectorVariable &rhs,
                      VectorVariable &solution,
                      const MPITopology3D &topo,
                      const DimensionsHandlerVector &dim_handler,
                      bool use_omp = false)
    {
        const Dim N = Nx;
        const Real h = dx;
        const Real h2 = h * h;

        // Domain length in x, consistent with your code
        const Real Lx = dx * (Nx - Real(0.5));

        // communicator along X
        MPI_Comm comm_x_raw = topo.comm_x();
        int rank_x = 0, size_x = 1;
        MPI_Comm_rank(comm_x_raw, &rank_x);
        MPI_Comm_size(comm_x_raw, &size_x);
        MPICommunicator comm_x(comm_x_raw, false);

        // component roles (normal/tangential) for X-sweep
        const Dim Comp1 = dim_handler.Comp1; // normal
        const Dim Comp2 = dim_handler.Comp2; // tangent
        const Dim Comp3 = dim_handler.Comp3; // tangent

        // constant gamma -> coefficients independent of (j,k)
        const Real gamma0 = gamma_field.get(0, 0, 0);
        const Real coeff0 = gamma0 / h2;

        // Two Schur solvers only (same as correct)
        SchurComplementSolver schur_dir(N, size_x, rank_x, comm_x);
        SchurComplementSolver schur_gho(N, size_x, rank_x, comm_x);

        const int local_N = schur_dir.get_local_N();
        const int global_start = schur_dir.get_global_start();

        // ------------------------------------------------------------
        // Build global (a,b,c) ONCE for Dirichlet-Dirichlet and Ghost
        // ------------------------------------------------------------
        std::vector<Real> a_dir(N, 0.0), b_dir(N, 0.0), c_dir(N, 0.0);
        std::vector<Real> a_gho(N, 0.0), b_gho(N, 0.0), c_gho(N, 0.0);

        // internal rows (same for both matrices)
        for (Dim i = 1; i < N - 1; ++i)
        {
            a_dir[i] = -coeff0;
            b_dir[i] = Real(1.0) + Real(2.0) * coeff0;
            c_dir[i] = -coeff0;

            a_gho[i] = -coeff0;
            b_gho[i] = Real(1.0) + Real(2.0) * coeff0;
            c_gho[i] = -coeff0;
        }

        // left boundary row: identity for both
        a_dir[0] = a_gho[0] = Real(0.0);
        b_dir[0] = b_gho[0] = Real(1.0);
        c_dir[0] = c_gho[0] = Real(0.0);

        // right boundary:
        // - Dirichlet (normal)
        a_dir[N - 1] = Real(0.0);
        b_dir[N - 1] = Real(1.0);
        c_dir[N - 1] = Real(0.0);

        // - Ghost elimination (tangentials): a=-coeff, b=1+3*coeff, c=0
        a_gho[N - 1] = -coeff0;
        b_gho[N - 1] = (Real(1.0) + Real(2.0) * coeff0) - (-coeff0); // 1 + 3*coeff
        c_gho[N - 1] = Real(0.0);

        // ------------------------------------------------------------
        // Preprocess ONCE per matrix
        // ------------------------------------------------------------
        auto preprocess = [&](SchurComplementSolver &schur,
                              const std::vector<Real> &a_full,
                              const std::vector<Real> &b_full,
                              const std::vector<Real> &c_full)
        {
            std::vector<Real> a_local(local_N, 0.0), b_local(local_N, 0.0), c_local(local_N, 0.0);
            for (int il = 0; il < local_N; ++il)
            {
                const int ig = global_start + il;
                if (ig < 0 || ig >= int(N))
                    continue;
                a_local[il] = a_full[ig];
                b_local[il] = b_full[ig];
                c_local[il] = c_full[ig];
            }
            schur.preprocess(a_local, b_local, c_local);
        };

        preprocess(schur_dir, a_dir, b_dir, c_dir);
        preprocess(schur_gho, a_gho, b_gho, c_gho);

        // ------------------------------------------------------------
        // Local (j,k) range
        // ------------------------------------------------------------
        Dim j0 = topo.local_j0(Ny), j1 = topo.local_j1(Ny);
        Dim k0 = topo.local_k0(Nz), k1 = topo.local_k1(Nz);

        const int nJ = int(j1 - j0);
        const int nK = int(k1 - k0);
        const int nLinesLocal = nJ * nK;

        auto lid_of = [&](Dim j, Dim k) -> int
        { return int((k - k0) * nJ + (j - j0)); };

        auto jk_of_lid = [&](int lid, Dim &j, Dim &k)
        {
            k = k0 + Dim(lid / nJ);
            j = j0 + Dim(lid - int(k - k0) * nJ);
        };

        // write-back: avoid duplicate left interface
        const int i0_local = (rank_x == 0) ? 0 : 1;

        // ------------------------------------------------------------
        // Build RHS only for local_N entries (FAST)
        // Boundary closure matches apply_bc<0>(rhs):
        //  - comp==direction (normal): u(0) - (dv/dy + dw/dz)*dx/2
        //  - comp tangential at left: pure Dirichlet
        //  - at right:
        //      normal: pure Dirichlet
        //      tangential: rhs(N-1) + 2*gamma/dx^2 * u_ex
        // ------------------------------------------------------------
        auto build_rhs_local = [&](Dim j, Dim k, Dim comp, Real *out)
        {
            for (int il = 0; il < local_N; ++il)
            {
                const int ig = global_start + il;

                if (ig < 0 || ig >= int(N))
                {
                    out[il] = Real(0.0);
                    continue;
                }

                // interior
                if (ig >= 1 && ig <= int(N) - 2)
                {
                    out[il] = rhs.value(comp, ig, j, k);
                    continue;
                }

                // boundary coordinates (match your apply_bc sampling)
                const Real y_u = Real(j) * dy;
                const Real y_v = (Real(j) + Real(0.5)) * dy;

                const Real z_u = Real(k) * dz;
                const Real z_w = (Real(k) + Real(0.5)) * dz;

                // ---------- left boundary ig==0 ----------
                if (ig == 0)
                {
                    if (comp == Comp1)
                    {
                        // NORMAL component: u(0,y,z) - (dv/dy + dw/dz)*dx/2
                        // use analytic derivatives (same as apply_bc)
                        const Real u0 = u_boundary.value<0>(Real(0.0), y_u, z_u, t);

                        // NOTE: these match apply_bc<0> calls:
                        // first_derivative<1> for dy, first_derivative<2> for dz
                        const Real dv_dy = u_boundary.first_derivative<1>(Real(0.0), y_u, z_u, t, dy);
                        const Real dw_dz = u_boundary.first_derivative<2>(Real(0.0), y_u, z_u, t, dz);

                        out[il] = u0 - (dv_dy + dw_dz) * (dx * Real(0.5));
                    }
                    else if (comp == Comp2)
                    {
                        // tangential v at x=0 : value at (0, y_v, z_u)
                        out[il] = u_boundary.value<1>(Real(0.0), y_v, z_u, t);
                    }
                    else // comp == Comp3
                    {
                        // tangential w at x=0 : value at (0, y_u, z_w)
                        out[il] = u_boundary.value<2>(Real(0.0), y_u, z_w, t);
                    }
                    continue;
                }

                // ---------- right boundary ig==N-1 ----------
                if (ig == int(N) - 1)
                {
                    if (comp == Comp1)
                    {
                        // NORMAL: Dirichlet at x=Lx, (Lx, y_u, z_u)
                        out[il] = u_boundary.value<0>(Lx, y_u, z_u, t);
                    }
                    else
                    {
                        // TANGENTIAL: rhs(N-1) + 2*gamma/dx^2 * u_ex
                        Real u_ex = Real(0.0);
                        if (comp == Comp2)
                            u_ex = u_boundary.value<1>(Lx, y_v, z_u, t);
                        else
                            u_ex = u_boundary.value<2>(Lx, y_u, z_w, t);

                        const Real gammaN = gamma_field.get(N - 1, j, k);
                        const Real coeffN = gammaN / h2;

                        out[il] = rhs.value(comp, N - 1, j, k) + Real(2.0) * coeffN * u_ex;
                    }
                    continue;
                }
            }
        };

        auto solve_component_batched = [&](Dim comp, SchurComplementSolver &schur)
        {
            std::vector<int> active_lids;
            active_lids.reserve(nLinesLocal);

            // known-face exactly like correct solver
            for (Dim k = k0; k < k1; ++k)
                for (Dim j = j0; j < j1; ++j)
                    if (!is_known_face<0>(j, k, comp))
                        active_lids.push_back(lid_of(j, k));
                    else
                        handle_known_face<0>(solution, j, k, comp);

            const int nActive = int(active_lids.size());
            if (nActive == 0)
                return;

            std::vector<Real> rhs_flat(size_t(nActive) * size_t(local_N));
            std::vector<Real> sol_flat(size_t(nActive) * size_t(local_N));

#pragma omp parallel for schedule(static) if (use_omp)
            for (int p = 0; p < nActive; ++p)
            {
                Dim j, k;
                jk_of_lid(active_lids[p], j, k);
                build_rhs_local(j, k, comp, rhs_flat.data() + size_t(p) * size_t(local_N));
            }

            schur.solve_batch(rhs_flat.data(), nActive, sol_flat.data(), use_omp);

#pragma omp parallel for schedule(static) if (use_omp)
            for (int p = 0; p < nActive; ++p)
            {
                Dim j, k;
                jk_of_lid(active_lids[p], j, k);

                const Real *x = sol_flat.data() + size_t(p) * size_t(local_N);
                for (int il = i0_local; il < local_N; ++il)
                {
                    const int ig = global_start + il;
                    if (ig < 0 || ig >= int(N))
                        continue;
                    solution.set(comp, ig, j, k) = x[il];
                }
            }
        };

        // normal uses Dirichlet matrix, tangentials use ghost matrix
        solve_component_batched(Comp1, schur_dir);
        solve_component_batched(Comp2, schur_gho);
        solve_component_batched(Comp3, schur_gho);
    }

    // =============================================================
    // Y-DIRECTION ONLY SOLVE (direction splitting validation)
    // Solves: (I - gamma * dyy) u = rhs   along y
    //
    // FAST VERSION:
    // - does NOT call apply_bc<1>(rhs)
    // - builds RHS locally per line (only local_N entries)
    // - boundary closure matches apply_bc<1> (analytic derivatives)
    // - known lines handled exactly like the correct solver
    // =============================================================
    void solve_y_only(VectorVariable &rhs,
                      VectorVariable &solution,
                      const MPITopology3D &topo,
                      const DimensionsHandlerVector &dim_handler,
                      bool use_omp = false)
    {
        const Dim N = Ny;
        const Real h = dy;
        const Real h2 = h * h;

        // Domain length in y
        const Real Ly = dy * (Ny - Real(0.5));

        // communicator along Y
        MPI_Comm comm_y_raw = topo.comm_y();
        int rank_y = 0, size_y = 1;
        MPI_Comm_rank(comm_y_raw, &rank_y);
        MPI_Comm_size(comm_y_raw, &size_y);
        MPICommunicator comm_y(comm_y_raw, false);

        // component roles (normal/tangential) for Y-sweep
        const Dim Comp1 = dim_handler.Comp1; // normal (direction==1)
        const Dim Comp2 = dim_handler.Comp2; // tangent
        const Dim Comp3 = dim_handler.Comp3; // tangent

        // constant gamma
        const Real gamma0 = gamma_field.get(0, 0, 0);
        const Real coeff0 = gamma0 / h2;

        // Two Schur solvers:
        SchurComplementSolver schur_dir(N, size_y, rank_y, comm_y);
        SchurComplementSolver schur_gho(N, size_y, rank_y, comm_y);

        const int local_N = schur_dir.get_local_N();
        const int global_start = schur_dir.get_global_start();

        // ------------------------------------------------------------
        // Build global (a,b,c) ONCE for Dirichlet-Dirichlet and Ghost
        // ------------------------------------------------------------
        std::vector<Real> a_dir(N, 0.0), b_dir(N, 0.0), c_dir(N, 0.0);
        std::vector<Real> a_gho(N, 0.0), b_gho(N, 0.0), c_gho(N, 0.0);

        for (Dim j = 1; j < N - 1; ++j)
        {
            a_dir[j] = -coeff0;
            b_dir[j] = Real(1.0) + Real(2.0) * coeff0;
            c_dir[j] = -coeff0;

            a_gho[j] = -coeff0;
            b_gho[j] = Real(1.0) + Real(2.0) * coeff0;
            c_gho[j] = -coeff0;
        }

        // left boundary row: identity for both
        a_dir[0] = a_gho[0] = Real(0.0);
        b_dir[0] = b_gho[0] = Real(1.0);
        c_dir[0] = c_gho[0] = Real(0.0);

        // right boundary:
        // - Dirichlet (normal)
        a_dir[N - 1] = Real(0.0);
        b_dir[N - 1] = Real(1.0);
        c_dir[N - 1] = Real(0.0);

        // - Ghost elimination (tangentials): a=-coeff, b=1+3*coeff, c=0
        a_gho[N - 1] = -coeff0;
        b_gho[N - 1] = (Real(1.0) + Real(2.0) * coeff0) - (-coeff0); // 1 + 3*coeff
        c_gho[N - 1] = Real(0.0);

        // preprocess once
        auto preprocess = [&](SchurComplementSolver &schur,
                              const std::vector<Real> &a_full,
                              const std::vector<Real> &b_full,
                              const std::vector<Real> &c_full)
        {
            std::vector<Real> a_local(local_N, 0.0), b_local(local_N, 0.0), c_local(local_N, 0.0);
            for (int il = 0; il < local_N; ++il)
            {
                const int jg = global_start + il;
                if (jg < 0 || jg >= int(N))
                    continue;
                a_local[il] = a_full[jg];
                b_local[il] = b_full[jg];
                c_local[il] = c_full[jg];
            }
            schur.preprocess(a_local, b_local, c_local);
        };

        preprocess(schur_dir, a_dir, b_dir, c_dir);
        preprocess(schur_gho, a_gho, b_gho, c_gho);

        // ------------------------------------------------------------
        // Local (i,k) range
        // ------------------------------------------------------------
        Dim i0 = topo.local_i0(Nx), i1 = topo.local_i1(Nx);
        Dim k0 = topo.local_k0(Nz), k1 = topo.local_k1(Nz);

        const int nI = int(i1 - i0);
        const int nK = int(k1 - k0);
        const int nLinesLocal = nI * nK;

        auto lid_of = [&](Dim i, Dim k) -> int
        { return int((k - k0) * nI + (i - i0)); };

        auto ik_of_lid = [&](int lid, Dim &i, Dim &k)
        {
            k = k0 + Dim(lid / nI);
            i = i0 + Dim(lid - int(k - k0) * nI);
        };

        // write-back: avoid duplicate left interface in Y
        const int j0_local = (rank_y == 0) ? 0 : 1;

        // ------------------------------------------------------------
        // Build RHS only for local_N entries (FAST)
        // Boundary closure matches apply_bc<1>(rhs):
        //  - comp==direction (normal): v(x,0,z) - (du/dx + dw/dz)*dy/2
        //  - tangential at left: pure Dirichlet
        //  - at right:
        //      normal: pure Dirichlet
        //      tangential: rhs(N-1) + 2*gamma/dy^2 * u_ex
        // ------------------------------------------------------------
        auto build_rhs_local = [&](Dim i, Dim k, Dim comp, Real *out)
        {
            for (int il = 0; il < local_N; ++il)
            {
                const int jg = global_start + il;

                if (jg < 0 || jg >= int(N))
                {
                    out[il] = Real(0.0);
                    continue;
                }

                // interior
                if (jg >= 1 && jg <= int(N) - 2)
                {
                    out[il] = rhs.value(comp, i, jg, k);
                    continue;
                }

                // coordinates (match apply_bc sampling)
                const Real x_v = Real(i) * dx;
                const Real x_u = (Real(i) + Real(0.5)) * dx;

                const Real z_v = Real(k) * dz;
                const Real z_w = (Real(k) + Real(0.5)) * dz;

                // ---------- left boundary jg==0 ----------
                if (jg == 0)
                {
                    if (comp == Comp1)
                    {
                        // NORMAL component (direction==1): v(x,0,z) - (du/dx + dw/dz)*dy/2
                        const Real v0 = u_boundary.value<1>(x_v, Real(0.0), z_v, t);

                        const Real du_dx = u_boundary.first_derivative<0>(x_v, Real(0.0), z_v, t, dx);
                        const Real dw_dz = u_boundary.first_derivative<2>(x_v, Real(0.0), z_v, t, dz);

                        out[il] = v0 - (du_dx + dw_dz) * (dy * Real(0.5));
                    }
                    else if (comp == Comp2)
                    {
                        // tangential u at y=0 : value at (x_u, 0, z_v)
                        out[il] = u_boundary.value<0>(x_u, Real(0.0), z_v, t);
                    }
                    else // comp == Comp3
                    {
                        // tangential w at y=0 : value at (x_v, 0, z_w)
                        out[il] = u_boundary.value<2>(x_v, Real(0.0), z_w, t);
                    }
                    continue;
                }

                // ---------- right boundary jg==N-1 ----------
                if (jg == int(N) - 1)
                {
                    if (comp == Comp1)
                    {
                        // NORMAL: Dirichlet at y=Ly, value at (x_v, Ly, z_v)
                        out[il] = u_boundary.value<1>(x_v, Ly, z_v, t);
                    }
                    else
                    {
                        // TANGENTIAL: rhs(N-1) + 2*gamma/dy^2 * u_ex
                        Real u_ex = Real(0.0);
                        if (comp == Comp2)
                            u_ex = u_boundary.value<0>(x_u, Ly, z_v, t);
                        else
                            u_ex = u_boundary.value<2>(x_v, Ly, z_w, t);

                        const Real gammaN = gamma_field.get(i, N - 1, k);
                        const Real coeffN = gammaN / h2;

                        out[il] = rhs.value(comp, i, N - 1, k) + Real(2.0) * coeffN * u_ex;
                    }
                    continue;
                }
            }
        };

        auto solve_component_batched = [&](Dim comp, SchurComplementSolver &schur)
        {
            std::vector<int> active_lids;
            active_lids.reserve(nLinesLocal);

            for (Dim k = k0; k < k1; ++k)
                for (Dim i = i0; i < i1; ++i)
                    if (!is_known_face<1>(i, k, comp))
                        active_lids.push_back(lid_of(i, k));
                    else
                        handle_known_face<1>(solution, i, k, comp);

            const int nActive = int(active_lids.size());
            if (nActive == 0)
                return;

            std::vector<Real> rhs_flat(size_t(nActive) * size_t(local_N));
            std::vector<Real> sol_flat(size_t(nActive) * size_t(local_N));

#pragma omp parallel for schedule(static) if (use_omp)
            for (int p = 0; p < nActive; ++p)
            {
                Dim i, k;
                ik_of_lid(active_lids[p], i, k);
                build_rhs_local(i, k, comp, rhs_flat.data() + size_t(p) * size_t(local_N));
            }

            schur.solve_batch(rhs_flat.data(), nActive, sol_flat.data(), use_omp);

#pragma omp parallel for schedule(static) if (use_omp)
            for (int p = 0; p < nActive; ++p)
            {
                Dim i, k;
                ik_of_lid(active_lids[p], i, k);

                const Real *x = sol_flat.data() + size_t(p) * size_t(local_N);
                for (int il = j0_local; il < local_N; ++il)
                {
                    const int jg = global_start + il;
                    if (jg < 0 || jg >= int(N))
                        continue;
                    solution.set(comp, i, jg, k) = x[il];
                }
            }
        };

        // normal uses Dirichlet matrix, tangentials use ghost matrix
        solve_component_batched(Comp1, schur_dir);
        solve_component_batched(Comp2, schur_gho);
        solve_component_batched(Comp3, schur_gho);
    }

    // =============================================================
    // Z-DIRECTION ONLY SOLVE (direction splitting validation)
    // Solves: (I - gamma * dzz) u = rhs   along z
    //
    // FAST VERSION:
    // - does NOT call apply_bc<2>(rhs)
    // - builds RHS locally per line (only local_N entries)
    // - boundary closure matches apply_bc<2> (analytic derivatives)
    // - known lines handled exactly like the correct solver
    // =============================================================
    void solve_z_only(VectorVariable &rhs,
                      VectorVariable &solution,
                      const MPITopology3D &topo,
                      const DimensionsHandlerVector &dim_handler,
                      bool use_omp = false)
    {
        const Dim N = Nz;
        const Real h = dz;
        const Real h2 = h * h;

        // Domain length in z
        const Real Lz = dz * (Nz - Real(0.5));

        // communicator along Z
        MPI_Comm comm_z_raw = topo.comm_z();
        int rank_z = 0, size_z = 1;
        MPI_Comm_rank(comm_z_raw, &rank_z);
        MPI_Comm_size(comm_z_raw, &size_z);
        MPICommunicator comm_z(comm_z_raw, false);

        // component roles (normal/tangential) for Z-sweep
        const Dim Comp1 = dim_handler.Comp1; // normal (direction==2)
        const Dim Comp2 = dim_handler.Comp2; // tangent
        const Dim Comp3 = dim_handler.Comp3; // tangent

        // constant gamma
        const Real gamma0 = gamma_field.get(0, 0, 0);
        const Real coeff0 = gamma0 / h2;

        // Two Schur solvers
        SchurComplementSolver schur_dir(N, size_z, rank_z, comm_z);
        SchurComplementSolver schur_gho(N, size_z, rank_z, comm_z);

        const int local_N = schur_dir.get_local_N();
        const int global_start = schur_dir.get_global_start();

        // ------------------------------------------------------------
        // Build global (a,b,c) ONCE for Dirichlet-Dirichlet and Ghost
        // ------------------------------------------------------------
        std::vector<Real> a_dir(N, 0.0), b_dir(N, 0.0), c_dir(N, 0.0);
        std::vector<Real> a_gho(N, 0.0), b_gho(N, 0.0), c_gho(N, 0.0);

        for (Dim k = 1; k < N - 1; ++k)
        {
            a_dir[k] = -coeff0;
            b_dir[k] = Real(1.0) + Real(2.0) * coeff0;
            c_dir[k] = -coeff0;

            a_gho[k] = -coeff0;
            b_gho[k] = Real(1.0) + Real(2.0) * coeff0;
            c_gho[k] = -coeff0;
        }

        // left boundary row: identity for both
        a_dir[0] = a_gho[0] = Real(0.0);
        b_dir[0] = b_gho[0] = Real(1.0);
        c_dir[0] = c_gho[0] = Real(0.0);

        // right boundary:
        // - Dirichlet (normal)
        a_dir[N - 1] = Real(0.0);
        b_dir[N - 1] = Real(1.0);
        c_dir[N - 1] = Real(0.0);

        // - Ghost elimination (tangentials): a=-coeff, b=1+3*coeff, c=0
        a_gho[N - 1] = -coeff0;
        b_gho[N - 1] = (Real(1.0) + Real(2.0) * coeff0) - (-coeff0); // 1 + 3*coeff
        c_gho[N - 1] = Real(0.0);

        // preprocess once
        auto preprocess = [&](SchurComplementSolver &schur,
                              const std::vector<Real> &a_full,
                              const std::vector<Real> &b_full,
                              const std::vector<Real> &c_full)
        {
            std::vector<Real> a_local(local_N, 0.0), b_local(local_N, 0.0), c_local(local_N, 0.0);
            for (int il = 0; il < local_N; ++il)
            {
                const int kg = global_start + il;
                if (kg < 0 || kg >= int(N))
                    continue;
                a_local[il] = a_full[kg];
                b_local[il] = b_full[kg];
                c_local[il] = c_full[kg];
            }
            schur.preprocess(a_local, b_local, c_local);
        };

        preprocess(schur_dir, a_dir, b_dir, c_dir);
        preprocess(schur_gho, a_gho, b_gho, c_gho);

        // ------------------------------------------------------------
        // Local (i,j) range
        // ------------------------------------------------------------
        Dim i0 = topo.local_i0(Nx), i1 = topo.local_i1(Nx);
        Dim j0 = topo.local_j0(Ny), j1 = topo.local_j1(Ny);

        const int nI = int(i1 - i0);
        const int nJ = int(j1 - j0);
        const int nLinesLocal = nI * nJ;

        auto lid_of = [&](Dim i, Dim j) -> int
        { return int((j - j0) * nI + (i - i0)); };

        auto ij_of_lid = [&](int lid, Dim &i, Dim &j)
        {
            j = j0 + Dim(lid / nI);
            i = i0 + Dim(lid - int(j - j0) * nI);
        };

        // write-back: avoid duplicate left interface in Z
        const int k0_local = (rank_z == 0) ? 0 : 1;

        // ------------------------------------------------------------
        // Build RHS only for local_N entries (FAST)
        // Boundary closure matches apply_bc<2>(rhs):
        //  - comp==direction (normal): w(x,y,0) - (du/dx + dv/dy)*dz/2
        //  - tangential at left: pure Dirichlet
        //  - at right:
        //      normal: pure Dirichlet
        //      tangential: rhs(N-1) + 2*gamma/dz^2 * u_ex
        // ------------------------------------------------------------
        auto build_rhs_local = [&](Dim i, Dim j, Dim comp, Real *out)
        {
            for (int il = 0; il < local_N; ++il)
            {
                const int kg = global_start + il;

                if (kg < 0 || kg >= int(N))
                {
                    out[il] = Real(0.0);
                    continue;
                }

                // interior
                if (kg >= 1 && kg <= int(N) - 2)
                {
                    out[il] = rhs.value(comp, i, j, kg);
                    continue;
                }

                // coordinates (match apply_bc sampling)
                const Real x_w = Real(i) * dx;
                const Real x_u = (Real(i) + Real(0.5)) * dx;

                const Real y_w = Real(j) * dy;
                const Real y_v = (Real(j) + Real(0.5)) * dy;

                // ---------- left boundary kg==0 ----------
                if (kg == 0)
                {
                    if (comp == Comp1)
                    {
                        // NORMAL component (direction==2): w(x,y,0) - (du/dx + dv/dy)*dz/2
                        const Real w0 = u_boundary.value<2>(x_w, y_w, Real(0.0), t);

                        const Real du_dx = u_boundary.first_derivative<0>(x_w, y_w, Real(0.0), t, dx);
                        const Real dv_dy = u_boundary.first_derivative<1>(x_w, y_w, Real(0.0), t, dy);

                        out[il] = w0 - (du_dx + dv_dy) * (dz * Real(0.5));
                    }
                    else if (comp == Comp2)
                    {
                        // tangential u at z=0 : value at (x_u, y_w, 0)
                        out[il] = u_boundary.value<0>(x_u, y_w, Real(0.0), t);
                    }
                    else // comp == Comp3
                    {
                        // tangential v at z=0 : value at (x_w, y_v, 0)
                        out[il] = u_boundary.value<1>(x_w, y_v, Real(0.0), t);
                    }
                    continue;
                }

                // ---------- right boundary kg==N-1 ----------
                if (kg == int(N) - 1)
                {
                    if (comp == Comp1)
                    {
                        // NORMAL: Dirichlet at z=Lz, value at (x_w, y_w, Lz)
                        out[il] = u_boundary.value<2>(x_w, y_w, Lz, t);
                    }
                    else
                    {
                        // TANGENTIAL: rhs(N-1) + 2*gamma/dz^2 * u_ex
                        Real u_ex = Real(0.0);
                        if (comp == Comp2)
                            u_ex = u_boundary.value<0>(x_u, y_w, Lz, t);
                        else
                            u_ex = u_boundary.value<1>(x_w, y_v, Lz, t);

                        const Real gammaN = gamma_field.get(i, j, N - 1);
                        const Real coeffN = gammaN / h2;

                        out[il] = rhs.value(comp, i, j, N - 1) + Real(2.0) * coeffN * u_ex;
                    }
                    continue;
                }
            }
        };

        auto solve_component_batched = [&](Dim comp, SchurComplementSolver &schur)
        {
            std::vector<int> active_lids;
            active_lids.reserve(nLinesLocal);

            for (Dim j = j0; j < j1; ++j)
                for (Dim i = i0; i < i1; ++i)
                    if (!is_known_face<2>(i, j, comp))
                        active_lids.push_back(lid_of(i, j));
                    else
                        handle_known_face<2>(solution, i, j, comp);

            const int nActive = int(active_lids.size());
            if (nActive == 0)
                return;

            std::vector<Real> rhs_flat(size_t(nActive) * size_t(local_N));
            std::vector<Real> sol_flat(size_t(nActive) * size_t(local_N));

#pragma omp parallel for schedule(static) if (use_omp)
            for (int p = 0; p < nActive; ++p)
            {
                Dim i, j;
                ij_of_lid(active_lids[p], i, j);
                build_rhs_local(i, j, comp, rhs_flat.data() + size_t(p) * size_t(local_N));
            }

            schur.solve_batch(rhs_flat.data(), nActive, sol_flat.data(), use_omp);

#pragma omp parallel for schedule(static) if (use_omp)
            for (int p = 0; p < nActive; ++p)
            {
                Dim i, j;
                ij_of_lid(active_lids[p], i, j);

                const Real *x = sol_flat.data() + size_t(p) * size_t(local_N);
                for (int il = k0_local; il < local_N; ++il)
                {
                    const int kg = global_start + il;
                    if (kg < 0 || kg >= int(N))
                        continue;
                    solution.set(comp, i, j, kg) = x[il];
                }
            }
        };

        // normal uses Dirichlet matrix, tangentials use ghost matrix
        solve_component_batched(Comp1, schur_dir);
        solve_component_batched(Comp2, schur_gho);
        solve_component_batched(Comp3, schur_gho);
    }
};
#endif // SOLVER_HPP
