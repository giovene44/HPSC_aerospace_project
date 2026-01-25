// MPITopology3D.hpp
#pragma once
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <array>

/**
 * @brief RAII wrapper for a 3D Cartesian MPI topology + 1D sub-communicators.
 *
 * Responsibilities:
 *  - Create a 3D Cartesian communicator (cart3d)
 *  - Provide coordinates (pz,py,px) and dims (Pz,Py,Px)
 *  - Create 1D sub-communicators for line-solves:
 *      comm_x: fixed (pz,py), varying px   -> size Px
 *      comm_y: fixed (pz,px), varying py   -> size Py
 *      comm_z: fixed (py,px), varying pz   -> size Pz
 *
 * IMPORTANT:
 *  - This class does NOT call MPI_Init/MPI_Finalize. Do that outside.
 *  - This class owns the communicators it creates and frees them in ~MPITopology3D().
 */
class MPITopology3D
{
public:
    MPITopology3D(MPI_Comm parent,
                  int Pz, int Py, int Px,
                  bool periodic_z = false,
                  bool periodic_y = false,
                  bool periodic_x = false,
                  bool reorder = true)
        : parent_(parent)
    {
        if (Pz <= 0 || Py <= 0 || Px <= 0)
        {
            throw std::runtime_error("MPITopology3D: dims must be > 0");
        }

        int parent_size = 0;
        MPI_Comm_size(parent_, &parent_size);
        if (parent_size != Pz * Py * Px)
        {
            throw std::runtime_error(
                "MPITopology3D: parent communicator size (" + std::to_string(parent_size) +
                ") != Pz*Py*Px (" + std::to_string(Pz * Py * Px) + ")");
        }

        dims_ = {Pz, Py, Px};
        periods_ = {periodic_z ? 1 : 0, periodic_y ? 1 : 0, periodic_x ? 1 : 0};

        // Create 3D cart communicator
        int dims_arr[3] = {dims_[0], dims_[1], dims_[2]};
        int periods_arr[3] = {periods_[0], periods_[1], periods_[2]};
        int reorder_int = reorder ? 1 : 0;

        MPI_Comm tmp_cart = MPI_COMM_NULL;
        int rc = MPI_Cart_create(parent_, 3, dims_arr, periods_arr, reorder_int, &tmp_cart);
        if (rc != MPI_SUCCESS || tmp_cart == MPI_COMM_NULL)
        {
            throw std::runtime_error("MPITopology3D: MPI_Cart_create failed");
        }
        cart3d_ = tmp_cart;

        // Rank + coords in cart
        MPI_Comm_rank(cart3d_, &cart_rank_);
        int coords_arr[3] = {0, 0, 0};
        MPI_Cart_coords(cart3d_, cart_rank_, 3, coords_arr);
        coords_ = {coords_arr[0], coords_arr[1], coords_arr[2]}; // (pz,py,px)

        // Create 1D sub-communicators
        // comm_x: keep x dim only
        {
            int remain[3] = {0, 0, 1};
            MPI_Comm tmp = MPI_COMM_NULL;
            rc = MPI_Cart_sub(cart3d_, remain, &tmp);
            if (rc != MPI_SUCCESS || tmp == MPI_COMM_NULL)
            {
                throw std::runtime_error("MPITopology3D: MPI_Cart_sub for comm_x failed");
            }
            comm_x_ = tmp;
            MPI_Comm_rank(comm_x_, &rank_x_);
            MPI_Comm_size(comm_x_, &size_x_);
        }
        // comm_y: keep y dim only
        {
            int remain[3] = {0, 1, 0};
            MPI_Comm tmp = MPI_COMM_NULL;
            rc = MPI_Cart_sub(cart3d_, remain, &tmp);
            if (rc != MPI_SUCCESS || tmp == MPI_COMM_NULL)
            {
                throw std::runtime_error("MPITopology3D: MPI_Cart_sub for comm_y failed");
            }
            comm_y_ = tmp;
            MPI_Comm_rank(comm_y_, &rank_y_);
            MPI_Comm_size(comm_y_, &size_y_);
        }
        // comm_z: keep z dim only
        {
            int remain[3] = {1, 0, 0};
            MPI_Comm tmp = MPI_COMM_NULL;
            rc = MPI_Cart_sub(cart3d_, remain, &tmp);
            if (rc != MPI_SUCCESS || tmp == MPI_COMM_NULL)
            {
                throw std::runtime_error("MPITopology3D: MPI_Cart_sub for comm_z failed");
            }
            comm_z_ = tmp;
            MPI_Comm_rank(comm_z_, &rank_z_);
            MPI_Comm_size(comm_z_, &size_z_);
        }
    }

    ~MPITopology3D()
    {
        // Free in reverse-ish order; safe even if some are MPI_COMM_NULL.
        if (comm_x_ != MPI_COMM_NULL)
            MPI_Comm_free(&comm_x_);
        if (comm_y_ != MPI_COMM_NULL)
            MPI_Comm_free(&comm_y_);
        if (comm_z_ != MPI_COMM_NULL)
            MPI_Comm_free(&comm_z_);
        if (cart3d_ != MPI_COMM_NULL)
            MPI_Comm_free(&cart3d_);
    }

    MPITopology3D(const MPITopology3D &) = delete;
    MPITopology3D &operator=(const MPITopology3D &) = delete;

    MPITopology3D(MPITopology3D &&other) noexcept
    {
        *this = std::move(other);
    }
    MPITopology3D &operator=(MPITopology3D &&other) noexcept
    {
        if (this == &other)
            return *this;

        parent_ = other.parent_;
        cart3d_ = other.cart3d_;
        comm_x_ = other.comm_x_;
        comm_y_ = other.comm_y_;
        comm_z_ = other.comm_z_;
        dims_ = other.dims_;
        periods_ = other.periods_;
        coords_ = other.coords_;
        cart_rank_ = other.cart_rank_;
        rank_x_ = other.rank_x_;
        rank_y_ = other.rank_y_;
        rank_z_ = other.rank_z_;
        size_x_ = other.size_x_;
        size_y_ = other.size_y_;
        size_z_ = other.size_z_;

        other.cart3d_ = MPI_COMM_NULL;
        other.comm_x_ = MPI_COMM_NULL;
        other.comm_y_ = MPI_COMM_NULL;
        other.comm_z_ = MPI_COMM_NULL;
        return *this;
    }

    // --- Accessors ---
    MPI_Comm cart_comm() const { return cart3d_; }
    MPI_Comm comm_x() const { return comm_x_; } // x-line communicator (fixed pz,py)
    MPI_Comm comm_y() const { return comm_y_; } // y-line communicator (fixed pz,px)
    MPI_Comm comm_z() const { return comm_z_; } // z-line communicator (fixed py,px)

    int Pz() const { return dims_[0]; }
    int Py() const { return dims_[1]; }
    int Px() const { return dims_[2]; }

    int pz() const { return coords_[0]; }
    int py() const { return coords_[1]; }
    int px() const { return coords_[2]; }

    int cart_rank() const { return cart_rank_; }

    int rank_in_x() const { return rank_x_; }
    int rank_in_y() const { return rank_y_; }
    int rank_in_z() const { return rank_z_; }

    int size_x() const { return size_x_; } // should equal Px
    int size_y() const { return size_y_; } // should equal Py
    int size_z() const { return size_z_; } // should equal Pz

    int size_total() const { return dims_[0] * dims_[1] * dims_[2]; }

    // --- Neighbor ranks in the 3D cart (MPI_PROC_NULL on boundaries if non-periodic) ---
    // Direction: 0=z, 1=y, 2=x (consistent with coords order here)
    int neighbor_minus_x() const { return neighbor_cart_(2, -1); }
    int neighbor_plus_x() const { return neighbor_cart_(2, +1); }
    int neighbor_minus_y() const { return neighbor_cart_(1, -1); }
    int neighbor_plus_y() const { return neighbor_cart_(1, +1); }
    int neighbor_minus_z() const { return neighbor_cart_(0, -1); }
    int neighbor_plus_z() const { return neighbor_cart_(0, +1); }

    // MPITopology3D.hpp (aggiungi dentro la classe, public:)
    struct Range
    {
        Dim begin;
        Dim end; // esclusivo
        Dim size() const { return end - begin; }
    };

    // Distribuzione bilanciata: primi "r" rank prendono (q+1), gli altri q
    static Range block_range(Dim Nglobal, int p, int P)
    {
        const Dim q = Nglobal / Dim(P);
        const Dim r = Nglobal % Dim(P);

        Range out;
        if (Dim(p) < r)
        {
            out.begin = Dim(p) * (q + 1);
            out.end = out.begin + (q + 1);
        }
        else
        {
            out.begin = r * (q + 1) + (Dim(p) - r) * q;
            out.end = out.begin + q;
        }
        return out;
    }

    // --- Range locali lungo Y e Z (usati per scegliere quali (j,k) linee risolve il rank) ---
    Range local_y_range(Dim Ny_global) const { return block_range(Ny_global, py(), Py()); }
    Range local_z_range(Dim Nz_global) const { return block_range(Nz_global, pz(), Pz()); }

    // Se vuoi anche X (solo se decomponi Nx lungo Px)
    Range local_x_range(Dim Nx_global) const { return block_range(Nx_global, px(), Px()); }

    // Comodità: singoli begin/end
    Dim local_j0(Dim Ny_global) const { return local_y_range(Ny_global).begin; }
    Dim local_j1(Dim Ny_global) const { return local_y_range(Ny_global).end; }

    Dim local_k0(Dim Nz_global) const { return local_z_range(Nz_global).begin; }
    Dim local_k1(Dim Nz_global) const { return local_z_range(Nz_global).end; }

    Dim local_i0(Dim Nx_global) const { return local_x_range(Nx_global).begin; }
    Dim local_i1(Dim Nx_global) const { return local_x_range(Nx_global).end; }

private:
    int neighbor_cart_(int dim, int disp) const
    {
        int src = MPI_PROC_NULL, dst = MPI_PROC_NULL;
        // MPI_Cart_shift returns src (rank at -disp) and dst (rank at +disp) for positive disp
        // We'll call with disp=1 and pick src/dst accordingly.
        int rc = MPI_Cart_shift(cart3d_, dim, 1, &src, &dst);
        if (rc != MPI_SUCCESS)
            return MPI_PROC_NULL;
        return (disp < 0) ? src : dst;
    }

private:
    MPI_Comm parent_ = MPI_COMM_NULL;
    MPI_Comm cart3d_ = MPI_COMM_NULL;
    MPI_Comm comm_x_ = MPI_COMM_NULL;
    MPI_Comm comm_y_ = MPI_COMM_NULL;
    MPI_Comm comm_z_ = MPI_COMM_NULL;

    std::array<int, 3> dims_{0, 0, 0}; // (Pz,Py,Px)
    std::array<int, 3> periods_{0, 0, 0};
    std::array<int, 3> coords_{0, 0, 0}; // (pz,py,px)

    int cart_rank_ = 0;

    int rank_x_ = 0, rank_y_ = 0, rank_z_ = 0;
    int size_x_ = 1, size_y_ = 1, size_z_ = 1;
};
