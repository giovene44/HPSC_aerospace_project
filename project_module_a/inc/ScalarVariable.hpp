#ifndef SCALARVARIABLE_HPP
#define SCALARVARIABLE_HPP

#include "Variables.hpp"
#include "BoundaryFunctions.hpp"
#include "DomainDecomposition.hpp"
#include <vector>
#include <iostream>
#include <stdexcept>

#ifdef USE_MPI
#include <mpi.h>
#endif

class ScalarVariable
{
public:
    /**
     * Constructor:
     * Initializes an empty scalar field of given dimensions
     * and fills it with zeros.
     */
    ScalarVariable(const Dim Nx, const Dim Ny, const Dim Nz, const Real dx_, const Real dy_, const Real dz_)
        : Nx(Nx), Ny(Ny), Nz(Nz), dx(dx_), dy(dy_), dz(dz_)
#ifdef USE_MPI
        , decomp(nullptr)
#endif
    {
        data.assign(Nx * Ny * Nz, Real(0));
    }

#ifdef USE_MPI
    /**
     * @brief Constructor with domain decomposition for MPI
     */
    ScalarVariable(const Dim Nx, const Dim Ny, const Dim Nz, 
                   const Real dx_, const Real dy_, const Real dz_,
                   DomainDecomposition* decomp_ptr)
        : Nx(Nx), Ny(Ny), Nz(Nz), dx(dx_), dy(dy_), dz(dz_), decomp(decomp_ptr)
    {
        data.assign(Nx * Ny * Nz, Real(0));
    }

    /**
     * @brief Set domain decomposition pointer
     */
    void set_decomposition(DomainDecomposition* decomp_ptr) {
        decomp = decomp_ptr;
    }
#endif

    ScalarVariable(const ScalarVariable &other)
        : Nx(other.Nx), Ny(other.Ny), Nz(other.Nz),
          data(other.data), dx(other.dx), dy(other.dy), dz(other.dz)
#ifdef USE_MPI
        , decomp(other.decomp)
#endif
    {
    }

    /**
     * Constructor 2:
     * Initializes the scalar field with an existing data vector.
     * The vector must have size Nx * Ny * Nz.
     */
    ScalarVariable(const Dim Nx, const Dim Ny, const Dim Nz, const std::vector<Real> &input_data)
        : Nx(Nx), Ny(Ny), Nz(Nz)
#ifdef USE_MPI
        , decomp(nullptr)
#endif
    {
        if (input_data.size() != static_cast<size_t>(Nx * Ny * Nz))
        {
            throw std::invalid_argument(
                "Error in ScalarVariable constructor: input_data size does not match Nx*Ny*Nz");
        }
        data = input_data;
    }

    // --- Operators ---
    ScalarVariable operator+(const ScalarVariable &other) const
    {
        ScalarVariable result(Nx, Ny, Nz, dx, dy, dz);
        for (Dim index = 0; index < Nx * Ny * Nz; ++index)
        {
            result.set(index) = this->get(index) + other.get(index);
        }
        return result;
    }

    ScalarVariable operator-(const ScalarVariable &other) const
    {
        ScalarVariable result(Nx, Ny, Nz, dx, dy, dz);
        for (Dim index = 0; index < Nx * Ny * Nz; ++index)
        {
            result.set(index) = this->get(index) - other.get(index);
        }
        return result;
    }

    ScalarVariable &operator+=(const ScalarVariable &other)
    {
        for (Dim index = 0; index < Nx * Ny * Nz; ++index)
        {
            this->set(index) += other.get(index);
        }
        return *this;
    }

    ScalarVariable &operator-=(const ScalarVariable &other)
    {
        for (Dim index = 0; index < Nx * Ny * Nz; ++index)
        {
            this->set(index) -= other.get(index);
        }
        return *this;
    }

    ScalarVariable &operator=(const ScalarVariable &other)
    {
        if (this == &other)
            return *this;

        if (Nx != other.Nx || Ny != other.Ny || Nz != other.Nz)
            throw std::runtime_error("ScalarVariable::operator=: dimension mismatch");

        data = other.data;
        dx = other.dx;
        dy = other.dy;
        dz = other.dz;
        return *this;
    }
    // --- Accessors ---
    Real &set(Dim i, Dim j, Dim k)
    {
        return data[i + j * Nx + k * Nx * Ny];
    }

    Real &set(Dim index)
    {
        return data[index];
    }

    /**
     * @brief Sets all elements in the scalar field to the specified value.
     * @param value The Real value to assign to all elements.
     */
    void set_all(Real value)
    {
        std::fill(data.begin(), data.end(), value);
    }

    void set_all(BoundaryFunctions &other, Real t)
    {
        for (Dim idx = 0; idx < Nx * Ny * Nz; ++idx)
        {
            Dim i = idx % Nx;
            Dim j = (idx / Nx) % Ny;
            Dim k = idx / (Nx * Ny);

            // Convert grid indices to physical coordinates
            Real x = i * dx;
            Real y = j * dy;
            Real z = k * dz;

            this->set(idx) = other.value<0>(x, y, z, t);
        }
    }

    Real get(Dim i, Dim j, Dim k) const
    {
        return data[i + j * Nx + k * Nx * Ny];
    }

    Real get(Dim index) const
    {
        return data[index];
    }

    // --- Gradients ---
    // --- Gradients (Centered Finite Difference - 2nd Order Interior) ---

    Real getGradient_x(Dim i, Dim j, Dim k) const
    {
        // Interior points (Second Order Centered)
        if (i < Nx - 1)
        {
            Real lhs = get(i, j, k);
            Real rhs = get(i + 1, j, k);
            return (rhs - lhs) / dx;
        }
        else 
            return 0.0; // g not used at the boundary!

    }

    Real getGradient_y(Dim i, Dim j, Dim k) const
    {
        // Interior points (Second Order Centered)
        if (j < Ny - 1)
        {
            Real lhs = get(i, j, k);
            Real rhs = get(i, j + 1, k);
            return (rhs - lhs) / dy;
        }
        // Bottom Boundary (Forward Difference - 1st Order)
        else
        {
            return 0.0; // g not used at the boundary!
        }
     
    }

    Real getGradient_z(Dim i, Dim j, Dim k) const
    {
        // Interior points (Second Order Centered)
        if (k < Nz - 1)
        {
            Real lhs = get(i, j, k);
            Real rhs = get(i, j, k + 1);
            return (rhs - lhs) / dz;
        }
        else
        {
            return 0.0; // g not used at the boundary!  
        }
    }

    Real getGradient_x(Dim index) const
    {
        Dim i = index % Nx;
        Dim j = (index / Nx) % Ny;
        Dim k = index / (Nx * Ny);
        return getGradient_x(i, j, k);
    }

    Real getGradient_y(Dim index) const
    {
        Dim i = index % Nx;
        Dim j = (index / Nx) % Ny;
        Dim k = index / (Nx * Ny);
        return getGradient_y(i, j, k);
    }

    Real getGradient_z(Dim index) const
    {
        Dim i = index % Nx;
        Dim j = (index / Nx) % Ny;
        Dim k = index / (Nx * Ny);
        return getGradient_z(i, j, k);
    }

    ScalarVariable getGradient_x() const
    {
        // Overloaded function to compute gradient for all elements
        ScalarVariable gradient(Nx, Ny, Nz, dx, dy, dz);
        for (Dim i = 0; i < Nx; ++i)
        {
            for (Dim j = 0; j < Ny; ++j)
            {
                for (Dim k = 0; k < Nz; ++k)
                {
                    gradient.set(i, j, k) = getGradient_x(i, j, k);
                }
            }
        }
        return gradient;
    }

    ScalarVariable getGradient_y() const
    {
        // Overloaded function to compute gradient for all elements
        ScalarVariable gradient(Nx, Ny, Nz, dx, dy, dz);
        for (Dim i = 0; i < Nx; ++i)
        {
            for (Dim j = 0; j < Ny; ++j)
            {
                for (Dim k = 0; k < Nz; ++k)
                {
                    gradient.set(i, j, k) = getGradient_y(i, j, k);
                }
            }
        }
        return gradient;
    }

    ScalarVariable getGradient_z() const
    {
        // Overloaded function to compute gradient for all elements
        ScalarVariable gradient(Nx, Ny, Nz, dx, dy, dz);
        for (Dim i = 0; i < Nx; ++i)
        {
            for (Dim j = 0; j < Ny; ++j)
            {
                for (Dim k = 0; k < Nz; ++k)
                {
                    gradient.set(i, j, k) = getGradient_z(i, j, k);
                }
            }
        }
        return gradient;
    }

    inline Dim get_Nx() const { return Nx; }
    inline Dim get_Ny() const { return Ny; }
    inline Dim get_Nz() const { return Nz; }
    // --- Utility ---
    size_t size() const { return data.size(); }
    const std::vector<Real> &getData() const { return data; }
    std::vector<Real> &getData() { return data; }

#ifdef USE_MPI
    /**
     * @brief Exchange halo (ghost) layers with neighboring processes
     * @param direction 0=x, 1=y, 2=z direction to exchange
     */
    void exchange_halos(Dim direction) {
        if (decomp == nullptr) return;

        auto [left_neighbor, right_neighbor] = decomp->get_neighbors(direction);
        MPI_Comm comm = decomp->get_cart_comm();

        // Determine the number of elements to send/receive
        Dim count = 0;
        if (direction == 0) {
            count = Ny * Nz;
        } else if (direction == 1) {
            count = Nx * Nz;
        } else if (direction == 2) {
            count = Nx * Ny;
        }

        std::vector<Real> send_left(count), send_right(count);
        std::vector<Real> recv_left(count), recv_right(count);

        // Pack data to send
        if (direction == 0) { // x-direction
            // Send leftmost plane to left, rightmost to right
            for (Dim j = 0; j < Ny; ++j) {
                for (Dim k = 0; k < Nz; ++k) {
                    send_left[j * Nz + k] = get(0, j, k);
                    send_right[j * Nz + k] = get(Nx - 1, j, k);
                }
            }
        } else if (direction == 1) { // y-direction
            for (Dim i = 0; i < Nx; ++i) {
                for (Dim k = 0; k < Nz; ++k) {
                    send_left[i * Nz + k] = get(i, 0, k);
                    send_right[i * Nz + k] = get(i, Ny - 1, k);
                }
            }
        } else if (direction == 2) { // z-direction
            for (Dim i = 0; i < Nx; ++i) {
                for (Dim j = 0; j < Ny; ++j) {
                    send_left[i * Ny + j] = get(i, j, 0);
                    send_right[i * Ny + j] = get(i, j, Nz - 1);
                }
            }
        }

        // Non-blocking communication
        MPI_Request requests[4];
        int req_count = 0;

        // Send to left, receive from left
        if (left_neighbor != MPI_PROC_NULL) {
            MPI_Isend(send_left.data(), count, MPI_FLOAT, left_neighbor, 0, comm, &requests[req_count++]);
            MPI_Irecv(recv_left.data(), count, MPI_FLOAT, left_neighbor, 1, comm, &requests[req_count++]);
        }

        // Send to right, receive from right
        if (right_neighbor != MPI_PROC_NULL) {
            MPI_Isend(send_right.data(), count, MPI_FLOAT, right_neighbor, 1, comm, &requests[req_count++]);
            MPI_Irecv(recv_right.data(), count, MPI_FLOAT, right_neighbor, 0, comm, &requests[req_count++]);
        }

        // Wait for all communications to complete
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

        // Note: In current implementation, ghost layers are not stored separately
        // This method can be extended to update ghost zones if needed
    }

    /**
     * @brief Gather distributed data to root process (rank 0)
     * @param global_data Output vector (only valid on rank 0)
     */
    void gather_to_root(std::vector<Real>& global_data) const {
        if (decomp == nullptr) {
            global_data = data;
            return;
        }

        int rank = decomp->get_rank();
        auto [Nx_global, Ny_global, Nz_global] = decomp->get_global_dimensions();
        
        if (rank == 0) {
            global_data.resize(Nx_global * Ny_global * Nz_global);
        }

        // Each rank sends its local data
        auto [i_start, j_start, k_start] = decomp->get_local_start_indices();
        auto [Nx_local, Ny_local, Nz_local] = decomp->get_local_dimensions();

        // Gather local data sizes and displacements
        int size = decomp->get_size();
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);

        int local_size = Nx_local * Ny_local * Nz_local;
        MPI_Gather(&local_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, decomp->get_cart_comm());

        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < size; ++i) {
                displs[i] = displs[i - 1] + recvcounts[i - 1];
            }
        }

        // Simple approach: use MPI_Gatherv for now
        // More sophisticated approach would preserve 3D structure
        std::vector<Real> temp_global;
        if (rank == 0) {
            temp_global.resize(Nx_global * Ny_global * Nz_global);
        }

        MPI_Gatherv(data.data(), local_size, MPI_FLOAT,
                    temp_global.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                    0, decomp->get_cart_comm());

        if (rank == 0) {
            // Rearrange data from processor order to spatial order
            // For now, simple copy (assumes contiguous layout)
            global_data = temp_global;
        }
    }

    /**
     * @brief Scatter global data from root to all processes
     * @param global_data Input vector (only valid on rank 0)
     */
    void scatter_from_root(const std::vector<Real>& global_data) {
        if (decomp == nullptr) {
            data = global_data;
            return;
        }

        int rank = decomp->get_rank();
        int size = decomp->get_size();
        auto [Nx_local, Ny_local, Nz_local] = decomp->get_local_dimensions();

        // Compute send counts and displacements
        std::vector<int> sendcounts(size);
        std::vector<int> displs(size);

        int local_size = Nx_local * Ny_local * Nz_local;
        MPI_Gather(&local_size, 1, MPI_INT, sendcounts.data(), 1, MPI_INT, 0, decomp->get_cart_comm());

        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < size; ++i) {
                displs[i] = displs[i - 1] + sendcounts[i - 1];
            }
        }

        data.resize(local_size);
        MPI_Scatterv(global_data.data(), sendcounts.data(), displs.data(), MPI_FLOAT,
                     data.data(), local_size, MPI_FLOAT,
                     0, decomp->get_cart_comm());
    }
#endif

private:
    const Dim Nx;
    const Dim Ny;
    const Dim Nz;
    std::vector<Real> data;
    Real dx, dy, dz;
#ifdef USE_MPI
    DomainDecomposition* decomp;
#endif
};

#endif // SCALARVARIABLE_HPP
