#ifndef NAVIER_STOKES_BRINKMAN_HPP
#define NAVIER_STOKES_BRINKMAN_HPP

/**
 * @file navier_stokes_brinkman.hpp
 * @brief Direction-splitting solver for the unsteady Stokes-Brinkman equations
 *
 *        du/dt - nu Lap(u) + (nu/k) u + grad(p) = f,   div(u) = 0
 *
 * on a MAC (staggered) grid, following F. Auteri, "Solving the momentum and pressure
 * equations" (HPSC Module A, Lecture 5) and Guermond & Minev's direction splitting.
 *
 * Grid (per direction, N nodes, spacing h = L / (N - 1/2)):
 *   pressure          p_i       at  i h,          i = 0..N-1
 *   normal velocity   u_{i+1/2} at (i + 1/2) h,   i = 0..N-1   (the last one lies on x = L)
 * so x = 0 passes through the pressure / tangential-velocity nodes and x = L through the
 * normal-velocity nodes.
 *
 * Time step n -> n+1 (Crank-Nicolson in delta form, beta = 1 + nu dt / (2k),
 * gamma = nu dt / (2 beta)):
 *   p*      = p^{n-1/2} + phi^{n-1/2}
 *   xi      = u^n + dt/beta [ f^{n+1/2} - grad p* - (nu/k) u^n + nu (Dxx eta^n + Dyy zeta^n + Dzz u^n) ]
 *   (I - gamma Dxx)(eta^{n+1}  - eta^n ) = xi       - eta^n
 *   (I - gamma Dyy)(zeta^{n+1} - zeta^n) = eta^{n+1} - zeta^n
 *   (I - gamma Dzz)(u^{n+1}    - u^n   ) = zeta^{n+1} - u^n
 *   (I - Dxx) psi = -div(u^{n+1}) / dt,  (I - Dyy) chi = psi,  (I - Dzz) phi^{n+1/2} = chi
 *   p^{n+1/2} = p^{n-1/2} + phi^{n+1/2}            (standard incremental form, chi = 0)
 *   p^{n+1/2} = p^{n-1/2} + phi^{n+1/2} - chi nu div(u^{n+1} + u^n)/2  (rotational, 0 < chi <= 1)
 * with homogeneous Neumann conditions for the pressure increments.
 *
 * Velocity boundary conditions (Dirichlet). For a velocity component and a direction d:
 *  - boundary through the nodes: the node value is prescribed;
 *  - boundary half a cell away from the last node: second-order ghost extrapolation
 *    u_ghost = 8/3 u_b - 2 u_0 + 1/3 u_1 ("Boundary Conditions Review", G. Martella).
 * Exactly the same discrete operator (same closures) is used in the explicit and in the
 * implicit part, so that the delta form is exact and the scheme is unconditionally stable;
 * the same ghost value is used to evaluate the divergence on the boundary pressure nodes.
 */

#include "Variables.hpp"
#include "Field3D.hpp"
#include "Parallel.hpp"
#include "TridiagonalSolver.hpp"
#include "Expression.hpp"
#include <array>
#include <string>
#include <vector>

struct ProblemData
{
    std::array<Real, 3> L{1, 1, 1};           // domain size
    Real nu = 1;                               // kinematic viscosity
    std::string k_expr = "1";                  // permeability k(x, y, z)
    std::array<std::string, 3> forcing;        // f(x, y, z, t)
    std::array<std::string, 3> u_exact;        // exact velocity (initial condition + errors)
    std::string p_exact = "0";                 // exact pressure (initial condition + errors)
    std::array<std::string, 3> u_boundary;     // Dirichlet boundary data for the velocity
    Real chi = 0;                              // rotational correction (0 = standard incremental)
};

struct ErrorNorms
{
    Real u_l2 = 0, u_l2_rel = 0, u_linf = 0;
    Real p_l2 = 0, p_l2_rel = 0, p_linf = 0;
    Real div_l2 = 0; // L2 norm of the discrete divergence of u
};

class NavierStokesBrinkman
{
public:
    NavierStokesBrinkman(const ProblemData &data, const Decomposition &dec, Real dt, Real t0);

    /// Set the exact solution at t0 (pressure at t0 - dt/2, staggered in time).
    void initialize();
    /// Advance one time step.
    void step();
    /// Advance nsteps time steps, returns the wall-clock time (max over the processes).
    double advance(int nsteps);

    /// Errors with respect to the exact solution at the current time.
    ErrorNorms errors() const;

    Real time() const { return t_; }
    Real dt() const { return dt_; }
    Real h(int d) const { return h_[d]; }

    /// Legacy-style VTK output of the cell-centred (interpolated) velocity and pressure
    /// (one file per process).
    void write_vtk(const std::string &prefix) const;

private:
    struct Face
    {
        bool active = false;
        int a = 0, b = 0, na = 0, nb = 0; // the two directions spanning the face
        std::vector<Real> v;
        Real at(int i, int j, int k) const
        {
            const int idx[3] = {i, j, k};
            return v[idx[a] + static_cast<std::size_t>(na) * idx[b]];
        }
    };

    const Decomposition &dec_;
    Real dt_, t_, nu_, chi_;
    std::array<Real, 3> L_, h_;
    std::array<int, 3> n_, g0_, N_;

    std::array<fastexpr::Expression, 3> f_, uex_, ubc_;
    fastexpr::Expression pex_, k_;

    // node coordinates: xc_[c][d] for velocity component c, xp_[d] for the pressure
    std::array<std::array<std::vector<Real>, 3>, 3> xc_;
    std::array<std::vector<Real>, 3> xp_;

    std::array<Field3D, 3> u_, eta_, zeta_, cs_, cu_;
    std::vector<std::vector<Real>> fbuf_; // per-thread forcing plane
    Real th_ = 0;                          // t^{n+1/2}
    Field3D p_, phi_, pstar_, psi_;
    Field3D div_new_, div_old_; // only for the rotational form (chi != 0)

    // boundary data: on-node Dirichlet values at t^{n+1}; half-cell values at t^n, t^{n+1}
    std::array<std::array<Face, 3>, 3> bon_, bhalf_old_, bhalf_new_;

    TridiagonalSolver solver_;
    bool uniform_v_ = false;                           // k constant: one matrix per sweep
    std::array<std::array<UniformMatrix, 3>, 3> umv_; // [direction][component]
    std::array<UniformMatrix, 3> ump_;                 // pressure, [direction]

    // helpers
    bool on_node_boundary(int c, int e, int G) const
    {
        return (e != c) ? (G == 0) : (G == N_[e] - 1);
    }
    int row_type(int c, int d, int G) const;
    void eval_field(const fastexpr::Expression &ex, int comp, Real t, Field3D &F) const;
    void eval_face(const fastexpr::Expression &ex, int c, int e, Real Xe, Real t, Face &F) const;
    void update_faces(Real t, std::array<std::array<Face, 3>, 3> &half, bool on_node);
    void fill_physical_ghost(int c, int d, Field3D &F, const std::array<std::array<Face, 3>, 3> &half) const;
    Real weighted_mean(const Field3D &F) const;
    Field3D output_pressure() const;
    /// Call f(i, j, k) for the local pressure nodes lying on the edges x=0/y=0, x=0/z=0 and
    /// y=0/z=0, which are not coupled to any velocity unknown.
    template <class F>
    void for_each_uncontrolled(F &&f) const
    {
        const bool z0[3] = {g0_[0] == 0, g0_[1] == 0, g0_[2] == 0};
        for (int d = 0; d < 3; ++d) // edge along direction d
        {
            const int a = (d + 1) % 3, b = (d + 2) % 3;
            if (!z0[a] || !z0[b])
                continue;
            for (int s = (d == 0 || !z0[d]) ? 0 : 1; s < n_[d]; ++s) // the corner only once
            {
                int idx[3];
                idx[d] = s;
                idx[a] = 0;
                idx[b] = 0;
                f(idx[0], idx[1], idx[2]);
            }
        }
    }

    template <int D>
    void velocity_sweep(int c);
    template <int D>
    void pressure_sweep(const Field3D *src, Field3D &dst);
};

#endif // NAVIER_STOKES_BRINKMAN_HPP
