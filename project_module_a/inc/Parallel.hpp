#ifndef PARALLEL_HPP
#define PARALLEL_HPP

/**
 * @file Parallel.hpp
 * @brief Cartesian process topology, block decomposition of the grid and ghost-cell
 *        exchange. Compiles both with and without MPI (USE_MPI): without MPI there is a
 *        single "process" and all communication functions are no-ops.
 */

#include "Variables.hpp"
#include "Field3D.hpp"
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

struct IndexRange
{
    Dim begin = 0;
    Dim end = 0; // exclusive
    Dim size() const { return end - begin; }
};

/// Balanced 1D block partition: the first (N % P) blocks get one extra point.
inline IndexRange block_partition(Dim N, int p, int P)
{
    const Dim q = N / P, r = N % P;
    IndexRange out;
    out.begin = p * q + (p < r ? p : r);
    out.end = out.begin + q + (p < r ? 1 : 0);
    return out;
}

/**
 * @brief 3D Cartesian process grid (Px x Py x Pz) with one line communicator per direction.
 */
class Topology
{
public:
#ifdef USE_MPI
    /// procs[d] <= 0 means "let MPI_Dims_create choose".
    explicit Topology(std::array<int, 3> procs = {0, 0, 0}, MPI_Comm parent = MPI_COMM_WORLD)
    {
        MPI_Comm_size(parent, &world_size_);
        int dims[3] = {procs[0] > 0 ? procs[0] : 0, procs[1] > 0 ? procs[1] : 0, procs[2] > 0 ? procs[2] : 0};
        if (dims[0] == 0 && dims[1] == 0 && dims[2] == 0)
        {
            // Prefer splitting z first, then y, then x: x-lines stay local as long as possible.
            int d[3] = {0, 0, 0};
            MPI_Dims_create(world_size_, 3, d); // non-increasing order
            dims[2] = d[0];
            dims[1] = d[1];
            dims[0] = d[2];
        }
        else
        {
            MPI_Dims_create(world_size_, 3, dims);
        }
        if (dims[0] * dims[1] * dims[2] != world_size_)
            throw std::runtime_error("Topology: Px*Py*Pz != number of MPI processes");
        int periods[3] = {0, 0, 0};
        // MPI orders the dims as given: we pass (x,y,z).
        MPI_Cart_create(parent, 3, dims, periods, 1, &cart_);
        MPI_Comm_rank(cart_, &rank_);
        int c[3];
        MPI_Cart_coords(cart_, rank_, 3, c);
        for (int d = 0; d < 3; ++d)
        {
            P_[d] = dims[d];
            coord_[d] = c[d];
            int remain[3] = {0, 0, 0};
            remain[d] = 1;
            MPI_Cart_sub(cart_, remain, &line_[d]);
            MPI_Cart_shift(cart_, d, 1, &nbr_[d][0], &nbr_[d][1]);
        }
    }
    ~Topology()
    {
        for (int d = 0; d < 3; ++d)
            if (line_[d] != MPI_COMM_NULL)
                MPI_Comm_free(&line_[d]);
        if (cart_ != MPI_COMM_NULL)
            MPI_Comm_free(&cart_);
    }
    MPI_Comm cart() const { return cart_; }
    MPI_Comm line(int d) const { return line_[d]; }
    int neighbor(int d, int side) const { return nbr_[d][side]; }
#else
    explicit Topology(std::array<int, 3> procs = {0, 0, 0})
    {
        for (int d = 0; d < 3; ++d)
            if (procs[d] > 1)
                throw std::runtime_error("Topology: built without MPI, only 1 process is supported");
    }
#endif
    Topology(const Topology &) = delete;
    Topology &operator=(const Topology &) = delete;

    int rank() const { return rank_; }
    int size() const { return world_size_; }
    int P(int d) const { return P_[d]; }
    int coord(int d) const { return coord_[d]; }

    // ---- reductions on the whole process grid
    double sum(double v) const
    {
#ifdef USE_MPI
        double r;
        MPI_Allreduce(&v, &r, 1, MPI_DOUBLE, MPI_SUM, cart_);
        return r;
#else
        return v;
#endif
    }
    void sum(double *v, int n) const
    {
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, cart_);
#else
        (void)v;
        (void)n;
#endif
    }
    double max(double v) const
    {
#ifdef USE_MPI
        double r;
        MPI_Allreduce(&v, &r, 1, MPI_DOUBLE, MPI_MAX, cart_);
        return r;
#else
        return v;
#endif
    }
    void barrier() const
    {
#ifdef USE_MPI
        MPI_Barrier(cart_);
#endif
    }

private:
    int world_size_ = 1;
    int rank_ = 0;
    std::array<int, 3> P_{1, 1, 1};
    std::array<int, 3> coord_{0, 0, 0};
#ifdef USE_MPI
    MPI_Comm cart_ = MPI_COMM_NULL;
    std::array<MPI_Comm, 3> line_{MPI_COMM_NULL, MPI_COMM_NULL, MPI_COMM_NULL};
    std::array<std::array<int, 2>, 3> nbr_{};
#endif
};

/**
 * @brief Block decomposition of a global grid of N[0] x N[1] x N[2] nodes on a Topology.
 *        All the staggered fields (u, v, w, p) have the same number of nodes per direction.
 */
class Decomposition
{
public:
    Decomposition(const Topology &topo, std::array<Dim, 3> Nglob) : topo_(topo), N_(Nglob)
    {
        for (int d = 0; d < 3; ++d)
        {
            range_[d] = block_partition(N_[d], topo.coord(d), topo.P(d));
            if (range_[d].size() < 2)
                throw std::runtime_error("Decomposition: every process needs at least 2 nodes per direction (N=" +
                                         std::to_string(N_[d]) + ", P=" + std::to_string(topo.P(d)) + ")");
        }
    }

    const Topology &topo() const { return topo_; }
    Dim N(int d) const { return N_[d]; }
    Dim n(int d) const { return range_[d].size(); }
    Dim begin(int d) const { return range_[d].begin; }
    Dim end(int d) const { return range_[d].end; }
    int P(int d) const { return topo_.P(d); }
    bool at_lo(int d) const { return topo_.coord(d) == 0; }
    bool at_hi(int d) const { return topo_.coord(d) == topo_.P(d) - 1; }

    /**
     * @brief Exchange the ghost layers of f along direction d with the two neighbours.
     *        Ghost layers on physical boundaries are left untouched.
     */
    void exchange(Field3D &f, int d) const
    {
#ifdef USE_MPI
        if (topo_.P(d) == 1)
            return;
        const int a = (d + 1) % 3, b = (d + 2) % 3;
        const int na = f.n(a), nb = f.n(b), nd = f.n(d);
        const long sa = f.stride(a), sb = f.stride(b), sd = f.stride(d);
        const std::size_t cnt = static_cast<std::size_t>(na) * nb;
        std::vector<Real> sbuf(2 * cnt), rbuf(2 * cnt);
        auto pack = [&](int layer, Real *buf)
        {
            const Real *base = f.raw() + f.idx(0, 0, 0) + layer * sd;
            for (int ib = 0; ib < nb; ++ib)
                for (int ia = 0; ia < na; ++ia)
                    buf[ia + static_cast<std::size_t>(na) * ib] = base[ia * sa + ib * sb];
        };
        auto unpack = [&](int layer, const Real *buf)
        {
            Real *base = f.raw() + f.idx(0, 0, 0) + layer * sd;
            for (int ib = 0; ib < nb; ++ib)
                for (int ia = 0; ia < na; ++ia)
                    base[ia * sa + ib * sb] = buf[ia + static_cast<std::size_t>(na) * ib];
        };
        const int lo = topo_.neighbor(d, 0), hi = topo_.neighbor(d, 1);
        pack(0, sbuf.data());              // first owned layer -> lo neighbour
        pack(nd - 1, sbuf.data() + cnt);   // last owned layer  -> hi neighbour
        MPI_Request req[4];
        int nreq = 0;
        if (lo != MPI_PROC_NULL)
        {
            MPI_Irecv(rbuf.data(), static_cast<int>(cnt), mpi_real_type(), lo, 100 + d, topo_.cart(), &req[nreq++]);
            MPI_Isend(sbuf.data(), static_cast<int>(cnt), mpi_real_type(), lo, 200 + d, topo_.cart(), &req[nreq++]);
        }
        if (hi != MPI_PROC_NULL)
        {
            MPI_Irecv(rbuf.data() + cnt, static_cast<int>(cnt), mpi_real_type(), hi, 200 + d, topo_.cart(), &req[nreq++]);
            MPI_Isend(sbuf.data() + cnt, static_cast<int>(cnt), mpi_real_type(), hi, 100 + d, topo_.cart(), &req[nreq++]);
        }
        MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
        if (lo != MPI_PROC_NULL)
            unpack(-1, rbuf.data());
        if (hi != MPI_PROC_NULL)
            unpack(nd, rbuf.data() + cnt);
#else
        (void)f;
        (void)d;
#endif
    }

private:
    const Topology &topo_;
    std::array<Dim, 3> N_;
    std::array<IndexRange, 3> range_;
};

#endif // PARALLEL_HPP
