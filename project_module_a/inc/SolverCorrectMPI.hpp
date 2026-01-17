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
    void solve_x_mpi(ScalarVariable &rhs,
                     ScalarVariable &solution,
                     const MPITopology3D &topo)
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
        // (I - dxx) with Neumann via ghost elimination like your serial code
        // ---------------------------
        std::vector<Real> a_full(N, -alpha);
        std::vector<Real> b_full(N, Real(1.0) + Real(2.0) * alpha);
        std::vector<Real> c_full(N, -alpha);

        // Neumann rows (ghost elimination)
        // left:  (1+2α) ψ0 - 2α ψ1 = rhs0 - 2 gL/h
        a_full[0] = Real(0.0);
        c_full[0] = -Real(2.0) * alpha;

        // right: -2α ψ_{N-2} + (1+2α) ψ_{N-1} = rhs_{N-1} + 2 gR/h
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
            int ig = global_start + il;
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
        Dim j0 = topo.local_j0(Ny);
        Dim j1 = topo.local_j1(Ny);
        Dim k0 = topo.local_k0(Nz);
        Dim k1 = topo.local_k1(Nz);

        // ---------------------------
        // Loop lines (j,k) and solve
        // ---------------------------
        for (Dim k = k0; k < k1; ++k)
            for (Dim j = j0; j < j1; ++j)
            {
                // Global RHS line d
                std::vector<Real> d(N, 0.0);

                for (Dim i = 0; i < N; ++i)
                    d[i] = rhs.get(i, j, k);

                // Neumann BC contributions (g = dψ/dx at boundary)
                const Real gL = p_boundary.value<0>(Real(0.0), j * dy, k * dz, t);
                const Real gR = p_boundary.value<0>((Nx - Real(0.5)) * dx, j * dy, k * dz, t);

                d[0] -= Real(2.0) * gL / h;
                d[N - 1] += Real(2.0) * gR / h;

                // Extract local RHS
                std::vector<Real> rhs_local(local_N, 0.0);
                for (int il = 0; il < local_N; ++il)
                {
                    int ig = global_start + il;
                    if (ig < 0 || ig >= (int)N)
                        continue;
                    rhs_local[il] = d[ig];
                }

                // Solve
                std::vector<Real> x_local;
                schur.solve(rhs_local, x_local);

                // Write back only owned part
                // (avoid duplicating the left interface point on ranks > 0)
                int i0_local = (rank_x == 0) ? 0 : 1;
                for (int il = i0_local; il < local_N; ++il)
                {
                    int ig = global_start + il;
                    if (ig < 0 || ig >= (int)N)
                        continue;
                    solution.set(ig, j, k) = x_local[il];
                }
            }
    }
    void solve_x(ScalarVariable &rhs, ScalarVariable &solution)
    {
        const Real alpha = Real(1.0) / (dx * dx); // α = 1/dx^2

        std::vector<Real> a(Nx, -alpha);
        std::vector<Real> b(Nx, Real(1.0) + Real(2.0) * alpha);
        std::vector<Real> c(Nx, -alpha);
        std::vector<Real> d(Nx), x(Nx);

        // Neumann rows via ghost elimination:
        // left:  (1+2α) ψ0 - 2α ψ1 = rhs0 - 2 g/dx
        a[0] = Real(0.0);
        c[0] = -Real(2.0) * alpha;

        // right: -2α ψ_{N-2} + (1+2α) ψ_{N-1} = rhs_{N-1} + 2 g/dx
        a[Nx - 1] = -Real(2.0) * alpha;
        c[Nx - 1] = Real(0.0);

        for (Dim j = 0; j < Ny; ++j)
            for (Dim k = 0; k < Nz; ++k)
            {
                // Copy rhs line
                for (Dim i = 0; i < Nx; ++i)
                    d[i] = rhs.get(i, j, k);

                // Add Neumann BC contributions to rhs (NOT inside matrix)
                // g = ∂ψ/∂x at boundary.
                const Real gL = p_boundary.value<0>(Real(0.0), j * dy, k * dz, t);
                const Real gR = p_boundary.value<0>((Nx - Real(0.5)) * dx, j * dy, k * dz, t);

                d[0] -= Real(2.0) * gL / dx;
                d[Nx - 1] += Real(2.0) * gR / dx;

                thomas_algorithm(a, b, c, d, x);

                for (Dim i = 0; i < Nx; ++i)
                    solution.set(i, j, k) = x[i];
            }
    }

    void solve_y(ScalarVariable &rhs, ScalarVariable &solution)
    {
        const Real alpha = Real(1.0) / (dy * dy);

        std::vector<Real> a(Ny, -alpha);
        std::vector<Real> b(Ny, Real(1.0) + Real(2.0) * alpha);
        std::vector<Real> c(Ny, -alpha);
        std::vector<Real> d(Ny), x(Ny);

        a[0] = Real(0.0);
        c[0] = -Real(2.0) * alpha;

        a[Ny - 1] = -Real(2.0) * alpha;
        c[Ny - 1] = Real(0.0);

        for (Dim i = 0; i < Nx; ++i)
            for (Dim k = 0; k < Nz; ++k)
            {
                for (Dim j = 0; j < Ny; ++j)
                    d[j] = rhs.get(i, j, k);

                const Real gB = p_boundary.value<1>(i * dx, Real(0.0), k * dz, t);
                const Real gT = p_boundary.value<1>(i * dx, (Ny - Real(0.5)) * dy, k * dz, t);

                d[0] -= Real(2.0) * gB / dy;
                d[Ny - 1] += Real(2.0) * gT / dy;

                thomas_algorithm(a, b, c, d, x);

                for (Dim j = 0; j < Ny; ++j)
                    solution.set(i, j, k) = x[j];
            }
    }

    void solve_y_mpi(ScalarVariable &rhs,
                     ScalarVariable &solution,
                     const MPITopology3D &topo)
    {
        const Dim N = Ny;
        const Real h = dy;
        const Real alpha = Real(1.0) / (h * h);

        // ---------------------------
        // Communicator in Y
        // ---------------------------
        MPI_Comm comm_y_raw = topo.comm_y();

        int rank_y = 0, size_y = 1;
        MPI_Comm_rank(comm_y_raw, &rank_y);
        MPI_Comm_size(comm_y_raw, &size_y);

        MPICommunicator comm_y(comm_y_raw, false);

        // ---------------------------
        // Build global a,b,c once (scalar)
        // (I - dyy) with Neumann via ghost elimination
        // ---------------------------
        std::vector<Real> a_full(N, -alpha);
        std::vector<Real> b_full(N, Real(1.0) + Real(2.0) * alpha);
        std::vector<Real> c_full(N, -alpha);

        // Neumann rows (ghost elimination)
        // bottom: (1+2α) ψ0 - 2α ψ1 = rhs0 - 2 gB/h
        a_full[0] = Real(0.0);
        c_full[0] = -Real(2.0) * alpha;

        // top: -2α ψ_{N-2} + (1+2α) ψ_{N-1} = rhs_{N-1} + 2 gT/h
        a_full[N - 1] = -Real(2.0) * alpha;
        c_full[N - 1] = Real(0.0);

        // ---------------------------
        // Schur solver preprocess (once)
        // ---------------------------
        SchurComplementSolver schur(N, size_y, rank_y, comm_y);

        const int local_N = schur.get_local_N();
        const int global_start = schur.get_global_start();

        std::vector<Real> a_local(local_N, 0.0), b_local(local_N, 0.0), c_local(local_N, 0.0);
        for (int il = 0; il < local_N; ++il)
        {
            int ig = global_start + il;
            if (ig < 0 || ig >= (int)N)
                continue;
            a_local[il] = a_full[ig];
            b_local[il] = b_full[ig];
            c_local[il] = c_full[ig];
        }
        schur.preprocess(a_local, b_local, c_local);

        // ---------------------------
        // Local (i,k) slab for this rank (in Px,Pz)
        // ---------------------------
        Dim i0 = topo.local_i0(Nx);
        Dim i1 = topo.local_i1(Nx);
        Dim k0 = topo.local_k0(Nz);
        Dim k1 = topo.local_k1(Nz);

        // ---------------------------
        // Loop lines (i,k) and solve
        // ---------------------------
        for (Dim k = k0; k < k1; ++k)
            for (Dim i = i0; i < i1; ++i)
            {
                // Global RHS line d
                std::vector<Real> d(N, 0.0);

                for (Dim j = 0; j < N; ++j)
                    d[j] = rhs.get(i, j, k);

                // Neumann BC contributions (g = dψ/dy at boundary)
                const Real gB = p_boundary.value<1>(i * dx, Real(0.0), k * dz, t);
                const Real gT = p_boundary.value<1>(i * dx, (Ny - Real(0.5)) * dy, k * dz, t);

                d[0] -= Real(2.0) * gB / h;
                d[N - 1] += Real(2.0) * gT / h;

                // Extract local RHS
                std::vector<Real> rhs_local(local_N, 0.0);
                for (int il = 0; il < local_N; ++il)
                {
                    int ig = global_start + il;
                    if (ig < 0 || ig >= (int)N)
                        continue;
                    rhs_local[il] = d[ig];
                }

                // Solve
                std::vector<Real> x_local;
                schur.solve(rhs_local, x_local);

                // Write back only owned part
                // (avoid duplicating the left interface point on ranks > 0)
                int j0_local = (rank_y == 0) ? 0 : 1;
                for (int il = j0_local; il < local_N; ++il)
                {
                    int ig = global_start + il;
                    if (ig < 0 || ig >= (int)N)
                        continue;
                    solution.set(i, ig, k) = x_local[il];
                }
            }
    }

    void solve_z(ScalarVariable &rhs, ScalarVariable &solution)
    {
        const Real alpha = Real(1.0) / (dz * dz);

        std::vector<Real> a(Nz, -alpha);
        std::vector<Real> b(Nz, Real(1.0) + Real(2.0) * alpha);
        std::vector<Real> c(Nz, -alpha);
        std::vector<Real> d(Nz), x(Nz);

        a[0] = Real(0.0);
        c[0] = -Real(2.0) * alpha;

        a[Nz - 1] = -Real(2.0) * alpha;
        c[Nz - 1] = Real(0.0);

        for (Dim i = 0; i < Nx; ++i)
            for (Dim j = 0; j < Ny; ++j)
            {
                for (Dim k = 0; k < Nz; ++k)
                    d[k] = rhs.get(i, j, k);

                const Real gF = p_boundary.value<2>(i * dx, j * dy, Real(0.0), t);
                const Real gB = p_boundary.value<2>(i * dx, j * dy, (Nz - Real(0.5)) * dz, t);

                d[0] -= Real(2.0) * gF / dz;
                d[Nz - 1] += Real(2.0) * gB / dz;

                thomas_algorithm(a, b, c, d, x);

                for (Dim k = 0; k < Nz; ++k)
                    solution.set(i, j, k) = x[k];
            }
    }

    void solve_z_mpi(ScalarVariable &rhs,
                     ScalarVariable &solution,
                     const MPITopology3D &topo)
    {
        const Dim N = Nz;
        const Real h = dz;
        const Real alpha = Real(1.0) / (h * h);

        // ---------------------------
        // Communicator in Z
        // ---------------------------
        MPI_Comm comm_z_raw = topo.comm_z();

        int rank_z = 0, size_z = 1;
        MPI_Comm_rank(comm_z_raw, &rank_z);
        MPI_Comm_size(comm_z_raw, &size_z);

        MPICommunicator comm_z(comm_z_raw, false);

        // ---------------------------
        // Build global a,b,c once (scalar)
        // (I - dzz) with Neumann via ghost elimination
        // ---------------------------
        std::vector<Real> a_full(N, -alpha);
        std::vector<Real> b_full(N, Real(1.0) + Real(2.0) * alpha);
        std::vector<Real> c_full(N, -alpha);

        // Neumann rows (ghost elimination)
        // front: (1+2α) ψ0 - 2α ψ1 = rhs0 - 2 gF/h
        a_full[0] = Real(0.0);
        c_full[0] = -Real(2.0) * alpha;

        // back: -2α ψ_{N-2} + (1+2α) ψ_{N-1} = rhs_{N-1} + 2 gB/h
        a_full[N - 1] = -Real(2.0) * alpha;
        c_full[N - 1] = Real(0.0);

        // ---------------------------
        // Schur solver preprocess (once)
        // ---------------------------
        SchurComplementSolver schur(N, size_z, rank_z, comm_z);

        const int local_N = schur.get_local_N();
        const int global_start = schur.get_global_start();

        std::vector<Real> a_local(local_N, 0.0), b_local(local_N, 0.0), c_local(local_N, 0.0);
        for (int il = 0; il < local_N; ++il)
        {
            int ig = global_start + il;
            if (ig < 0 || ig >= (int)N)
                continue;
            a_local[il] = a_full[ig];
            b_local[il] = b_full[ig];
            c_local[il] = c_full[ig];
        }
        schur.preprocess(a_local, b_local, c_local);

        // ---------------------------
        // Local (i,j) slab for this rank (in Px,Py)
        // ---------------------------
        Dim i0 = topo.local_i0(Nx);
        Dim i1 = topo.local_i1(Nx);
        Dim j0 = topo.local_j0(Ny);
        Dim j1 = topo.local_j1(Ny);

        // ---------------------------
        // Loop lines (i,j) and solve
        // ---------------------------
        for (Dim j = j0; j < j1; ++j)
            for (Dim i = i0; i < i1; ++i)
            {
                // Global RHS line d
                std::vector<Real> d(N, 0.0);

                for (Dim k = 0; k < N; ++k)
                    d[k] = rhs.get(i, j, k);

                // Neumann BC contributions (g = dψ/dz at boundary)
                const Real gF = p_boundary.value<2>(i * dx, j * dy, Real(0.0), t);
                const Real gB = p_boundary.value<2>(i * dx, j * dy, (Nz - Real(0.5)) * dz, t);

                d[0] -= Real(2.0) * gF / h;
                d[N - 1] += Real(2.0) * gB / h;

                // Extract local RHS
                std::vector<Real> rhs_local(local_N, 0.0);
                for (int il = 0; il < local_N; ++il)
                {
                    int ig = global_start + il;
                    if (ig < 0 || ig >= (int)N)
                        continue;
                    rhs_local[il] = d[ig];
                }

                // Solve
                std::vector<Real> x_local;
                schur.solve(rhs_local, x_local);

                // Write back only owned part
                // (avoid duplicating the left interface point on ranks > 0)
                int k0_local = (rank_z == 0) ? 0 : 1;
                for (int il = k0_local; il < local_N; ++il)
                {
                    int ig = global_start + il;
                    if (ig < 0 || ig >= (int)N)
                        continue;
                    solution.set(i, j, ig) = x_local[il];
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

    // =============================================================
    // X-DIRECTION ONLY SOLVE (direction splitting validation)
    // Solves: (I - gamma * dxx) u = rhs   along x
    // =============================================================

    void solve_x_only(VectorVariable &rhs,
                      VectorVariable &solution, const MPITopology3D &topo, bool use_omp = false)
    {
        const Dim N = Nx;
        const Real h = dx;
        const Real h2 = h * h;
        const Real Lx = dx * (Nx - Real(0.5));

        // ============================================================
        // 0) Communicator in X e rank/size
        // ============================================================
        MPI_Comm comm_x_raw = topo.comm_x();

        int rank_x = 0, size_x = 1;
        MPI_Comm_rank(comm_x_raw, &rank_x);
        MPI_Comm_size(comm_x_raw, &size_x);

        // Create an MPICommunicator view wrapper around the MPI_Comm
        // duplicate=false means we don't own the communicator lifecycle
        MPICommunicator comm_x(comm_x_raw, false);

        // ============================================================
        // 1) Costruisci (a,b,c) UNA VOLTA per componente (u,v,w)
        //    (nel test gamma è costante e non dipende da j,k)
        // ============================================================
        auto build_abc_for_comp = [&](int comp,
                                      std::vector<Real> &a,
                                      std::vector<Real> &b,
                                      std::vector<Real> &c)
        {
            a.assign(N, 0.0);
            b.assign(N, 0.0);
            c.assign(N, 0.0);

            // interni
            for (Dim i = 1; i < N - 1; ++i)
            {
                const Real gamma_val = gamma_field.get(i, 0, 0); // test: costante in (j,k)
                const Real coeff = gamma_val / h2;
                a[i] = -coeff;
                b[i] = Real(1.0) + Real(2.0) * coeff;
                c[i] = -coeff;
            }

            // sinistra: riga Dirichlet fissa (come nel tuo codice)
            a[0] = 0.0;
            b[0] = 1.0;
            c[0] = 0.0;

            // destra
            if (comp == 0)
            {
                // u: Dirichlet
                a[N - 1] = 0.0;
                b[N - 1] = 1.0;
                c[N - 1] = 0.0;
            }
            else
            {
                // v,w: ghost elimination
                const Real gammaN = gamma_field.get(N - 1, 0, 0);
                const Real coeff = gammaN / h2;
                a[N - 1] = -coeff;
                b[N - 1] = (Real(1.0) + Real(2.0) * coeff) - (-coeff); // 1 + 3*coeff
                c[N - 1] = 0.0;
            }
        };

        // ============================================================
        // 2) BC note per alcune linee (come nel tuo codice)
        // ============================================================
        auto is_known_face_x = [&](Dim j, Dim k, int comp) -> bool
        {
            if (comp == 0)
                return (j == 0 || k == 0);
            if (comp == 1)
                return (j == Ny - 1 || k == 0);
            return (j == 0 || k == Nz - 1);
        };

        auto fill_known_face_x = [&](Dim j, Dim k, int comp)
        {
            const Real y_u = Real(j) * dy;
            const Real y_v = (Real(j) + Real(0.5)) * dy;

            const Real z_w = (Real(k) + Real(0.5)) * dz;
            const Real z_u = Real(k) * dz;

            for (Dim i = 0; i < Nx; ++i)
            {
                const Real x_u = (Real(i) + Real(0.5)) * dx;
                const Real x_vw = Real(i) * dx;

                if (comp == 0)
                    solution.set(0, i, j, k) = u_boundary.value<0>(x_u, y_u, z_u, t);
                else if (comp == 1)
                    solution.set(1, i, j, k) = u_boundary.value<1>(x_vw, y_v, z_u, t);
                else
                    solution.set(2, i, j, k) = u_boundary.value<2>(x_vw, y_u, z_w, t);
            }
        };

        // ============================================================
        // 3) Crea 3 solver Schur (uno per componente) e preprocess UNA VOLTA
        // ============================================================
        SchurComplementSolver schur_u(N, size_x, rank_x, comm_x);
        SchurComplementSolver schur_v(N, size_x, rank_x, comm_x);
        SchurComplementSolver schur_w(N, size_x, rank_x, comm_x);

        const int local_N = schur_u.get_local_N();
        const int global_start = schur_u.get_global_start();

        std::vector<Real> a_full, b_full, c_full;

        auto preprocess_solver = [&](SchurComplementSolver &schur, int comp)
        {
            build_abc_for_comp(comp, a_full, b_full, c_full);

            std::vector<Real> a_local(local_N, 0.0), b_local(local_N, 0.0), c_local(local_N, 0.0);
            for (int il = 0; il < local_N; ++il)
            {
                int ig = global_start + il;
                if (ig < 0 || ig >= N)
                    continue;
                a_local[il] = a_full[ig];
                b_local[il] = b_full[ig];
                c_local[il] = c_full[ig];
            }
            schur.preprocess(a_local, b_local, c_local);
        };

        preprocess_solver(schur_u, 0);
        preprocess_solver(schur_v, 1);
        preprocess_solver(schur_w, 2);

        // ============================================================
        // 4) Solve per linea: costruisci SOLO d (RHS globale della linea)
        //    poi estrai rhs_local, solve, e scrivi SOLO owned part
        // ============================================================
        auto solve_line_for_component =
            [&](Dim j, Dim k, int comp, SchurComplementSolver &schur)
        {
            if (is_known_face_x(j, k, comp))
            {
                fill_known_face_x(j, k, comp);
                return;
            }

            // d = RHS globale della linea
            std::vector<Real> d(N, 0.0);

            // -------------------------
            // interni
            // -------------------------
            for (Dim i = 1; i < N - 1; ++i)
                d[i] = rhs.value(comp, i, j, k);

            // -------------------------
            // bordo sinistro i=0
            // (qui metti ESATTAMENTE la tua logica)
            // -------------------------
            {
                const Real y = j * dy;
                const Real z = k * dz;

                if (comp == 0)
                {
                    const Real u0_wall = u_boundary.value<0>(Real(0.0), y, z, t);

                    const Real v_plus = u_boundary.value<1>(Real(0.0), (Real(j) + Real(0.5)) * dy, z, t);
                    const Real v_minus = u_boundary.value<1>(Real(0.0), (Real(j) - Real(0.5)) * dy, z, t);
                    const Real dv_dy = (v_plus - v_minus) / dy;

                    const Real w_plus = u_boundary.value<2>(Real(0.0), y, (Real(k) + Real(0.5)) * dz, t);
                    const Real w_minus = u_boundary.value<2>(Real(0.0), y, (Real(k) - Real(0.5)) * dz, t);
                    const Real dw_dz = (w_plus - w_minus) / dz;

                    const Real dudx0 = -(dv_dy + dw_dz);
                    const Real u_half = u0_wall + dudx0 * (dx * Real(0.5));
                    d[0] = u_half;
                }
                else if (comp == 1)
                {
                    d[0] = u_boundary.value<1>(Real(0.0), (Real(j) + Real(0.5)) * dy, z, t);
                }
                else
                {
                    d[0] = u_boundary.value<2>(Real(0.0), y, (Real(k) + Real(0.5)) * dz, t);
                }
            }

            // -------------------------
            // bordo destro i=N-1
            // (anche qui la tua logica)
            // -------------------------
            if (comp == 0)
            {
                const Real y = j * dy;
                const Real z = k * dz;
                d[N - 1] = u_boundary.value<0>(Lx, y, z, t);
            }
            else
            {
                const Real y = j * dy;
                const Real z = k * dz;

                Real u_ex = 0.0;
                if (comp == 1)
                    u_ex = u_boundary.value<1>(Lx, (Real(j) + Real(0.5)) * dy, z, t);
                else
                    u_ex = u_boundary.value<2>(Lx, y, (Real(k) + Real(0.5)) * dz, t);

                // Nota: il termine +2*coeff*u_ex era già nel tuo schema per i=N-1
                const Real gammaN = gamma_field.get(N - 1, j, k);
                const Real coeff = gammaN / h2;
                d[N - 1] = rhs.value(comp, N - 1, j, k) + Real(2.0) * coeff * u_ex;
            }

            // ------------------------------------------------------------
            // RHS locale per Schur
            // ------------------------------------------------------------
            std::vector<Real> rhs_local(local_N, 0.0);
            for (int il = 0; il < local_N; ++il)
            {
                int ig = global_start + il;
                if (ig < 0 || ig >= N)
                    continue;
                rhs_local[il] = d[ig];
            }

            // solve locale (Schur fa comunicazione solo in comm_x)
            std::vector<Real> x_local;
            schur.solve(rhs_local, x_local);

            // write-back: SOLO porzione owned (evita duplicato interfaccia sinistra)
            int i0_local = (rank_x == 0) ? 0 : 1;
            for (int il = i0_local; il < local_N; ++il)
            {
                int ig = global_start + il;
                if (ig < 0 || ig >= N)
                    continue;
                solution.set(comp, ig, j, k) = x_local[il];
            }
        };

        // ============================================================
        // 5) Loop locale su (j,k) del process (Py,Pz)
        // ============================================================
        Dim j0 = topo.local_j0(Ny);
        Dim j1 = topo.local_j1(Ny);
        Dim k0 = topo.local_k0(Nz);
        Dim k1 = topo.local_k1(Nz);

        for (Dim k = k0; k < k1; ++k)
            for (Dim j = j0; j < j1; ++j)
            {
                solve_line_for_component(j, k, 0, schur_u);
                solve_line_for_component(j, k, 1, schur_v);
                solve_line_for_component(j, k, 2, schur_w);
            }
    }

    // =============================================================
    // Y-DIRECTION ONLY SOLVE (direction splitting validation)
    // Solves: (I - gamma * dyy) u = rhs   along y
    // =============================================================
    void solve_y_only(VectorVariable &rhs,
                      VectorVariable &solution,
                      const MPITopology3D &topo,
                      bool use_omp = false)
    {
        const Dim N = Ny;
        const Real h = dy;
        const Real h2 = h * h;

        const Real Ly = dy * (Ny - Real(0.5));

        // ============================================================
        // 0) Communicator in Y e rank/size
        // ============================================================
        MPI_Comm comm_y_raw = topo.comm_y();
        int rank_y = 0, size_y = 1;
        MPI_Comm_rank(comm_y_raw, &rank_y);
        MPI_Comm_size(comm_y_raw, &size_y);

        // Create an MPICommunicator view wrapper around the MPI_Comm
        MPICommunicator comm_y(comm_y_raw, false);

        auto build_abc_for_comp = [&](int comp,
                                      std::vector<Real> &a,
                                      std::vector<Real> &b,
                                      std::vector<Real> &c)
        {
            a.assign(N, 0.0);
            b.assign(N, 0.0);
            c.assign(N, 0.0);

            // Intern
            for (Dim j = 1; j < N - 1; ++j)
            {
                const Real gamma_val = gamma_field.get(0, j, 0); // test: costante in (i,k)
                const Real coeff = gamma_val / h2;
                a[j] = -coeff;
                b[j] = Real(1.0) + Real(2.0) * coeff;
                c[j] = -coeff;
            }

            // Left boundary: Dirichlet fixed row (as in your code)
            {
                a[0] = 0.0;
                b[0] = 1.0;
                c[0] = 0.0;
            }
            // Right boundary
            {
                if (comp == 1)
                {
                    // v: Dirichlet
                    a[N - 1] = 0.0;
                    b[N - 1] = 1.0;
                    c[N - 1] = 0.0;
                }
                else
                {
                    // u,w: ghost elimination
                    const Real gammaN = gamma_field.get(0, N - 1, 0);
                    const Real coeff = gammaN / h2;
                    a[N - 1] = -coeff;
                    b[N - 1] = (Real(1.0) + Real(2.0) * coeff) - (-coeff); // 1 + 3*coeff
                    c[N - 1] = 0.0;
                }
            }
        };

        // ---- Known face logic for direction = 1 (matches your earlier is_known_face<1>) ----
        auto is_known_face_y = [&](Dim i, Dim k, int comp) -> bool
        {
            if (comp == 0)
                return (i == Nx - 1 || k == 0);
            if (comp == 1)
                return (i == 0 || k == 0);
            // comp == 2
            return (i == 0 || k == Nz - 1);
        };

        // Fill ONLY the requested component along the whole y-line at (i,k)
        auto fill_known_face_y = [&](Dim i, Dim k, int comp)
        {
            const Real x_u = (Real(i) + Real(0.5)) * dx; // u location in x
            const Real x_vw = Real(i) * dx;              // v,w location in x

            const Real z_u = Real(k) * dz;
            const Real z_w = (Real(k) + Real(0.5)) * dz;

            for (Dim j = 0; j < Ny; ++j)
            {
                const Real y_u = Real(j) * dy;               // u,w location in y
                const Real y_v = (Real(j) + Real(0.5)) * dy; // v location in y

                if (comp == 0)
                {
                    // u at (x+dx/2, y, z)
                    solution.set(0, i, j, k) = u_boundary.value<0>(x_u, y_u, z_u, t);
                }
                else if (comp == 1)
                {
                    // v at (x, y+dy/2, z)
                    solution.set(1, i, j, k) = u_boundary.value<1>(x_vw, y_v, z_u, t);
                }
                else
                {
                    // w at (x, y, z+dz/2)
                    solution.set(2, i, j, k) = u_boundary.value<2>(x_vw, y_u, z_w, t);
                }
            }
        };

        // ============================================================
        // 3) Crea 3 solver Schur (uno per componente) e preprocess UNA VOLTA
        // ============================================================
        SchurComplementSolver schur_u(N, size_y, rank_y, comm_y);
        SchurComplementSolver schur_v(N, size_y, rank_y, comm_y);
        SchurComplementSolver schur_w(N, size_y, rank_y, comm_y);

        const int local_N = schur_u.get_local_N();
        const int global_start = schur_u.get_global_start();
        std::vector<Real> a_full, b_full, c_full;

        auto preprocess_solver = [&](SchurComplementSolver &schur, int comp)
        {
            build_abc_for_comp(comp, a_full, b_full, c_full);

            std::vector<Real> a_local(local_N, 0.0), b_local(local_N, 0.0), c_local(local_N, 0.0);
            for (int il = 0; il < local_N; ++il)
            {
                int ig = global_start + il;
                if (ig < 0 || ig >= N)
                    continue;
                a_local[il] = a_full[ig];
                b_local[il] = b_full[ig];
                c_local[il] = c_full[ig];
            }
            schur.preprocess(a_local, b_local, c_local);
        };

        preprocess_solver(schur_u, 0);
        preprocess_solver(schur_v, 1);
        preprocess_solver(schur_w, 2);

        std::vector<Real> d(N, 0.0);
        auto solve_line_for_component = [&](Dim i, Dim k, int comp, SchurComplementSolver &schur)
        {
            // Known face: skip TDMA
            if (is_known_face_y(i, k, comp))
            {
                fill_known_face_y(i, k, comp);
                return;
            }

            // -------------------------
            // Interior coefficients
            // -------------------------
            for (Dim j = 1; j < N - 1; ++j)
                d[j] = rhs.value(comp, i, j, k);

            // =========================================================
            // LEFT boundary (y=0): lecture BCs
            // Normal comp=1 (v): incompressibility reconstruction
            // Tangentials (u,w): Dirichlet
            // =========================================================
            {

                const Real x_u = (Real(i) + Real(0.5)) * dx;
                const Real x_vw = Real(i) * dx;
                const Real z_u = Real(k) * dz;
                const Real z_w = (Real(k) + Real(0.5)) * dz;

                if (comp == 1)
                {
                    // v is stored at y = dy/2 -> that's v_{1/2}
                    // v_{1/2} = v(0) + (dv/dy)|0 * dy/2
                    // (dv/dy)|0 = -(du/dx)|0 - (dw/dz)|0
                    const Real v0_wall = u_boundary.value<1>(x_vw, Real(0.0), z_u, t);

                    const Real u_plus = u_boundary.value<0>(x_u + Real(0.5) * dx, Real(0.0), z_u, t);
                    const Real u_minus = u_boundary.value<0>(x_u - Real(0.5) * dx, Real(0.0), z_u, t);
                    const Real du_dx = (u_plus - u_minus) / dx;

                    const Real w_plus = u_boundary.value<2>(x_vw, Real(0.0), z_w + Real(0.5) * dz, t);
                    const Real w_minus = u_boundary.value<2>(x_vw, Real(0.0), z_w - Real(0.5) * dz, t);
                    const Real dw_dz = (w_plus - w_minus) / dz;

                    const Real dvdy0 = -(du_dx + dw_dz);

                    const Real v_half = v0_wall + dvdy0 * (dy * Real(0.5));
                    d[0] = v_half;
                }
                else if (comp == 0)
                {
                    // u at y=0 is on boundary
                    d[0] = u_boundary.value<0>(x_u, Real(0.0), z_u, t);
                }
                else // comp == 2
                {
                    // w at y=0 is on boundary
                    d[0] = u_boundary.value<2>(x_vw, Real(0.0), z_w, t);
                }
            }

            // =========================================================
            // RIGHT boundary (y=Ly): lecture BCs
            // Normal comp=1 (v): Dirichlet at j=N-1 (v located at y=Ly)
            // Tangentials (u,w): ghost elimination at j=N-1
            // =========================================================
            if (comp == 1)
            {

                const Real x_vw = Real(i) * dx;
                const Real z_u = Real(k) * dz;
                d[N - 1] = u_boundary.value<1>(x_vw, Ly, z_u, t);
            }
            else
            {
                // ghost elimination on last row:
                // a u_{N-2} + (b - c) u_{N-1} = rhs + 2*coeff*u_ex
                const Real gammaN = gamma_field.get(i, N - 1, k);
                const Real coeff = gammaN / h2;

                const Real x_u = (Real(i) + Real(0.5)) * dx;
                const Real x_vw = Real(i) * dx;
                const Real z_u = Real(k) * dz;
                const Real z_w = (Real(k) + Real(0.5)) * dz;

                Real u_ex = 0.0;
                if (comp == 0)
                    u_ex = u_boundary.value<0>(x_u, Ly, z_u, t);
                else
                    u_ex = u_boundary.value<2>(x_vw, Ly, z_w, t);

                d[N - 1] = rhs.value(comp, i, N - 1, k) + Real(2.0) * coeff * u_ex;
            }

            // ------------------------------------------------------------
            // RHS locale per Schur
            // ------------------------------------------------------------
            std::vector<Real> rhs_local(local_N, 0.0);
            for (int il = 0; il < local_N; ++il)
            {
                int ig = global_start + il;
                if (ig < 0 || ig >= N)
                    continue;
                rhs_local[il] = d[ig];
            }

            // solve locale (Schur fa comunicazione solo in comm_y)
            std::vector<Real> x_local;
            schur.solve(rhs_local, x_local);

            // Write back solo porzione owned (evita duplicato interfaccia sinistra)
            int j0_local = (rank_y == 0) ? 0 : 1;
            for (int il = j0_local; il < local_N; ++il)
            {
                int ig = global_start + il;
                if (ig < 0 || ig >= N)
                    continue;
                solution.set(comp, i, ig, k) = x_local[il];
            }
        };

        // ============================================================
        // 5) Loop locale su (i,k) del process (Px,Pz)
        // ============================================================
        Dim i0 = topo.local_i0(Nx);
        Dim i1 = topo.local_i1(Nx);
        Dim k0 = topo.local_k0(Nz);
        Dim k1 = topo.local_k1(Nz);
        for (Dim k = k0; k < k1; ++k)
            for (Dim i = i0; i < i1; ++i)
            {
                solve_line_for_component(i, k, 0, schur_u);
                solve_line_for_component(i, k, 1, schur_v);
                solve_line_for_component(i, k, 2, schur_w);
            }
    }

    // =============================================================
    // Z-DIRECTION ONLY SOLVE (direction splitting validation)
    // Solves: (I - gamma * dzz) u = rhs   along z
    // =============================================================
    void solve_z_only(VectorVariable &rhs,
                      VectorVariable &solution,
                      const MPITopology3D &topo,
                      bool use_omp = false)
    {
        const Dim N = Nz;
        const Real h = dz;
        const Real h2 = h * h;

        const Real Lz = dz * (Nz - Real(0.5));

        // ============================================================
        // 0) Communicator in Z e rank/size
        // ============================================================
        MPI_Comm comm_z_raw = topo.comm_z();
        int rank_z = 0, size_z = 1;
        MPI_Comm_rank(comm_z_raw, &rank_z);
        MPI_Comm_size(comm_z_raw, &size_z);

        // Create an MPICommunicator view wrapper around the MPI_Comm
        MPICommunicator comm_z(comm_z_raw, false);

        auto build_abc_for_comp = [&](int comp,
                                      std::vector<Real> &a,
                                      std::vector<Real> &b,
                                      std::vector<Real> &c)
        {
            a.assign(N, 0.0);
            b.assign(N, 0.0);
            c.assign(N, 0.0);

            // Intern
            for (Dim k = 1; k < N - 1; ++k)
            {
                const Real gamma_val = gamma_field.get(0, 0, k); // test: costante in (i,j)
                const Real coeff = gamma_val / h2;
                a[k] = -coeff;
                b[k] = Real(1.0) + Real(2.0) * coeff;
                c[k] = -coeff;
            }

            // Left boundary: Dirichlet fixed row (as in your code)
            {
                a[0] = 0.0;
                b[0] = 1.0;
                c[0] = 0.0;
            }
            // Right boundary
            {
                if (comp == 2)
                {
                    // w: Dirichlet
                    a[N - 1] = 0.0;
                    b[N - 1] = 1.0;
                    c[N - 1] = 0.0;
                }
                else
                {
                    // u,v: ghost elimination
                    const Real gammaN = gamma_field.get(0, 0, N - 1);
                    const Real coeff = gammaN / h2;
                    a[N - 1] = -coeff;
                    b[N - 1] = (Real(1.0) + Real(2.0) * coeff) - (-coeff); // 1 + 3*coeff
                    c[N - 1] = 0.0;
                }
            }
        };

        // ---- Known face logic for direction = 2 (matches your earlier is_known_face<2>) ----
        // Here index_1 = i, index_2 = j
        auto is_known_face_z = [&](Dim i, Dim j, int comp) -> bool
        {
            if (comp == 0)
                return (i == Nx - 1 || j == 0);
            if (comp == 1)
                return (i == 0 || j == Ny - 1);
            // comp == 2
            return (i == 0 || j == 0);
        };

        // Fill ONLY the requested component along the whole z-line at (i,j)
        auto fill_known_face_z = [&](Dim i, Dim j, int comp)
        {
            const Real x_u = (Real(i) + Real(0.5)) * dx; // u location in x
            const Real x_vw = Real(i) * dx;              // v,w location in x

            const Real y_u = Real(j) * dy;               // u,w location in y
            const Real y_v = (Real(j) + Real(0.5)) * dy; // v location in y

            for (Dim k = 0; k < Nz; ++k)
            {
                const Real z_u = Real(k) * dz;               // u,v location in z
                const Real z_w = (Real(k) + Real(0.5)) * dz; // w location in z

                if (comp == 0)
                {
                    // u at (x+dx/2, y, z)
                    solution.set(0, i, j, k) = u_boundary.value<0>(x_u, y_u, z_u, t);
                }
                else if (comp == 1)
                {
                    // v at (x, y+dy/2, z)
                    solution.set(1, i, j, k) = u_boundary.value<1>(x_vw, y_v, z_u, t);
                }
                else
                {
                    // w at (x, y, z+dz/2)
                    solution.set(2, i, j, k) = u_boundary.value<2>(x_vw, y_u, z_w, t);
                }
            }
        };

        // ============================================================
        // 3) Crea 3 solver Schur (uno per componente) e preprocess
        // ============================================================
        SchurComplementSolver schur_u(N, size_z, rank_z, comm_z);
        SchurComplementSolver schur_v(N, size_z, rank_z, comm_z);
        SchurComplementSolver schur_w(N, size_z, rank_z, comm_z);
        const int local_N = schur_u.get_local_N();
        const int global_start = schur_u.get_global_start();
        std::vector<Real> a_full, b_full, c_full;

        auto preprocess_solver = [&](SchurComplementSolver &schur, int comp)
        {
            build_abc_for_comp(comp, a_full, b_full, c_full);

            std::vector<Real> a_local(local_N, 0.0), b_local(local_N, 0.0), c_local(local_N, 0.0);
            for (int il = 0; il < local_N; ++il)
            {
                int ig = global_start + il;
                if (ig < 0 || ig >= N)
                    continue;
                a_local[il] = a_full[ig];
                b_local[il] = b_full[ig];
                c_local[il] = c_full[ig];
            }
            schur.preprocess(a_local, b_local, c_local);
        };

        preprocess_solver(schur_u, 0);
        preprocess_solver(schur_v, 1);
        preprocess_solver(schur_w, 2);
        std::vector<Real> d(N, 0.0);

        auto solve_line_for_component = [&](Dim i, Dim j, int comp, SchurComplementSolver &schur)
        {
            // Known face: skip TDMA
            if (is_known_face_z(i, j, comp))
            {
                fill_known_face_z(i, j, comp);
                return;
            }

            // -------------------------
            // Interior coefficients
            // -------------------------
            for (Dim k = 1; k < N - 1; ++k)
            {
                const Real gamma_val = gamma_field.get(i, j, k);
                const Real coeff = gamma_val / h2;

                d[k] = rhs.value(comp, i, j, k);
            }

            // =========================================================
            // LEFT boundary (z=0): lecture BCs
            // Normal comp=2 (w): incompressibility reconstruction
            // Tangentials (u,v): Dirichlet
            // =========================================================
            {

                const Real x_u = (Real(i) + Real(0.5)) * dx;
                const Real x_vw = Real(i) * dx;

                const Real y_u = Real(j) * dy;
                const Real y_v = (Real(j) + Real(0.5)) * dy;

                if (comp == 2)
                {
                    // w is stored at z = dz/2 -> that's w_{1/2}
                    // w_{1/2} = w(0) + (dw/dz)|0 * dz/2
                    // (dw/dz)|0 = -(du/dx)|0 - (dv/dy)|0
                    const Real w0_wall = u_boundary.value<2>(x_vw, y_u, Real(0.0), t);

                    const Real u_plus = u_boundary.value<0>(x_u + Real(0.5) * dx, y_u, Real(0.0), t);
                    const Real u_minus = u_boundary.value<0>(x_u - Real(0.5) * dx, y_u, Real(0.0), t);
                    const Real du_dx = (u_plus - u_minus) / dx;

                    const Real v_plus = u_boundary.value<1>(x_vw, y_v + Real(0.5) * dy, Real(0.0), t);
                    const Real v_minus = u_boundary.value<1>(x_vw, y_v - Real(0.5) * dy, Real(0.0), t);
                    const Real dv_dy = (v_plus - v_minus) / dy;

                    const Real dwdz0 = -(du_dx + dv_dy);

                    const Real w_half = w0_wall + dwdz0 * (dz * Real(0.5));
                    d[0] = w_half;
                }
                else if (comp == 0)
                {
                    // u at z=0 is on boundary
                    d[0] = u_boundary.value<0>(x_u, y_u, Real(0.0), t);
                }
                else // comp == 1
                {
                    // v at z=0 is on boundary
                    d[0] = u_boundary.value<1>(x_vw, y_v, Real(0.0), t);
                }
            }

            // =========================================================
            // RIGHT boundary (z=Lz): lecture BCs
            // Normal comp=2 (w): Dirichlet at k=N-1 (w located at z=Lz)
            // Tangentials (u,v): ghost elimination at k=N-1
            // =========================================================
            if (comp == 2)
            {
                const Real x_vw = Real(i) * dx;
                const Real y_u = Real(j) * dy;
                d[N - 1] = u_boundary.value<2>(x_vw, y_u, Lz, t);
            }
            else
            {
                const Real gammaN = gamma_field.get(i, j, N - 1);
                const Real coeff = gammaN / h2;

                const Real x_u = (Real(i) + Real(0.5)) * dx;
                const Real x_vw = Real(i) * dx;

                const Real y_u = Real(j) * dy;
                const Real y_v = (Real(j) + Real(0.5)) * dy;

                Real u_ex = 0.0;
                if (comp == 0)
                    u_ex = u_boundary.value<0>(x_u, y_u, Lz, t);
                else
                    u_ex = u_boundary.value<1>(x_vw, y_v, Lz, t);

                d[N - 1] = rhs.value(comp, i, j, N - 1) + Real(2.0) * coeff * u_ex;
            }

            // ------------------------------------------------------------
            // RHS locale per Schur
            // ------------------------------------------------------------
            std::vector<Real> rhs_local(local_N, 0.0);
            for (int il = 0; il < local_N; ++il)
            {
                int ig = global_start + il;
                if (ig < 0 || ig >= N)
                    continue;
                rhs_local[il] = d[ig];
            }

            // solve locale (Schur fa comunicazione solo in comm_z)
            std::vector<Real> x_local;
            schur.solve(rhs_local, x_local);

            // Write back solo porzione owned (evita duplicato interfaccia sinistra)
            int k0_local = (rank_z == 0) ? 0 : 1;
            for (int il = k0_local; il < local_N; ++il)
            {
                int ig = global_start + il;
                if (ig < 0 || ig >= N)
                    continue;
                solution.set(comp, i, j, ig) = x_local[il];
            }
        };

        // ============================================================
        // 5) Loop locale su (i,j) del process (Px,Py)
        // ============================================================
        Dim i0 = topo.local_i0(Nx);
        Dim i1 = topo.local_i1(Nx);
        Dim j0 = topo.local_j0(Ny);
        Dim j1 = topo.local_j1(Ny);
        for (Dim j = j0; j < j1; ++j)
            for (Dim i = i0; i < i1; ++i)
            {
                solve_line_for_component(i, j, 0, schur_u);
                solve_line_for_component(i, j, 1, schur_v);
                solve_line_for_component(i, j, 2, schur_w);
            }
    }
};
#endif // SOLVER_HPP
