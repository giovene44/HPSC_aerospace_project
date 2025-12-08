#ifndef DOMAINDECOMPOSITION_HPP
#define DOMAINDECOMPOSITION_HPP

#include "Variables.hpp"
#include <array>
#include <stdexcept>
#include <cmath>
#include <algorithm>

#ifdef USE_MPI
#include <mpi.h>
#endif

// Enum for boundary sides
enum class BoundarySide { LEFT = 0, RIGHT = 1 };

// Structure to hold local partition information
struct LocalPartition {
    Dim Nx_local, Ny_local, Nz_local;  // Local dimensions
    Dim i_start, j_start, k_start;      // Global start indices
    Dim i_end, j_end, k_end;            // Global end indices
    std::array<int, 6> neighbors;       // Neighbor ranks [x-,x+,y-,y+,z-,z+]
    std::array<bool, 6> has_boundary;   // Physical boundary flags [x-,x+,y-,y+,z-,z+]
};

class DomainDecomposition {
public:
    #ifdef USE_MPI
    /**
     * @brief Constructor for MPI domain decomposition
     * @param Nx_global Global grid size in x
     * @param Ny_global Global grid size in y
     * @param Nz_global Global grid size in z
     * @param comm MPI communicator
     * @param Px_user User-specified partitions in x (-1 for automatic)
     * @param Py_user User-specified partitions in y (-1 for automatic)
     * @param Pz_user User-specified partitions in z (-1 for automatic)
     */
    DomainDecomposition(Dim Nx_global, Dim Ny_global, Dim Nz_global,
                        MPI_Comm comm = MPI_COMM_WORLD,
                        int Px_user = -1, int Py_user = -1, int Pz_user = -1)
        : Nx_global(Nx_global), Ny_global(Ny_global), Nz_global(Nz_global),
          comm_world(comm)
    {
        MPI_Comm_rank(comm_world, &rank);
        MPI_Comm_size(comm_world, &size);

        // Determine process grid dimensions
        compute_process_grid(Px_user, Py_user, Pz_user);

        // Create Cartesian communicator
        create_cartesian_grid();

        // Setup local partition for this rank
        setup_local_partition();
    }

    /**
     * @brief Get local dimensions for current rank
     */
    std::array<Dim, 3> get_local_dimensions() const {
        return {local_partition.Nx_local, local_partition.Ny_local, local_partition.Nz_local};
    }

    /**
     * @brief Get local start indices in global coordinates
     */
    std::array<Dim, 3> get_local_start_indices() const {
        return {local_partition.i_start, local_partition.j_start, local_partition.k_start};
    }

    /**
     * @brief Get local end indices in global coordinates (exclusive)
     */
    std::array<Dim, 3> get_local_end_indices() const {
        return {local_partition.i_end, local_partition.j_end, local_partition.k_end};
    }

    /**
     * @brief Convert global indices to local indices
     * @return true if point is in local partition, false otherwise
     */
    bool global_to_local(Dim i_global, Dim j_global, Dim k_global,
                         Dim& i_local, Dim& j_local, Dim& k_local) const {
        if (i_global >= local_partition.i_start && i_global < local_partition.i_end &&
            j_global >= local_partition.j_start && j_global < local_partition.j_end &&
            k_global >= local_partition.k_start && k_global < local_partition.k_end) {
            i_local = i_global - local_partition.i_start;
            j_local = j_global - local_partition.j_start;
            k_local = k_global - local_partition.k_start;
            return true;
        }
        return false;
    }

    /**
     * @brief Convert local indices to global indices
     */
    void local_to_global(Dim i_local, Dim j_local, Dim k_local,
                         Dim& i_global, Dim& j_global, Dim& k_global) const {
        i_global = i_local + local_partition.i_start;
        j_global = j_local + local_partition.j_start;
        k_global = k_local + local_partition.k_start;
    }

    /**
     * @brief Check if a point is an interface point in a given direction
     * @param i_local Local i index
     * @param j_local Local j index
     * @param k_local Local k index
     * @param direction 0=x, 1=y, 2=z
     * @return true if point is at interface boundary
     */
    bool is_interface_point(Dim i_local, Dim j_local, Dim k_local, Dim direction) const {
        if (direction == 0) { // x-direction
            return (i_local == 0 && local_partition.neighbors[0] != MPI_PROC_NULL) ||
                   (i_local == local_partition.Nx_local - 1 && local_partition.neighbors[1] != MPI_PROC_NULL);
        } else if (direction == 1) { // y-direction
            return (j_local == 0 && local_partition.neighbors[2] != MPI_PROC_NULL) ||
                   (j_local == local_partition.Ny_local - 1 && local_partition.neighbors[3] != MPI_PROC_NULL);
        } else if (direction == 2) { // z-direction
            return (k_local == 0 && local_partition.neighbors[4] != MPI_PROC_NULL) ||
                   (k_local == local_partition.Nz_local - 1 && local_partition.neighbors[5] != MPI_PROC_NULL);
        }
        return false;
    }

    /**
     * @brief Get neighbor ranks for a given direction
     * @param direction 0=x, 1=y, 2=z
     * @return pair of (left_neighbor, right_neighbor) ranks (MPI_PROC_NULL if none)
     */
    std::pair<int, int> get_neighbors(Dim direction) const {
        if (direction == 0) {
            return {local_partition.neighbors[0], local_partition.neighbors[1]};
        } else if (direction == 1) {
            return {local_partition.neighbors[2], local_partition.neighbors[3]};
        } else if (direction == 2) {
            return {local_partition.neighbors[4], local_partition.neighbors[5]};
        }
        return {MPI_PROC_NULL, MPI_PROC_NULL};
    }

    /**
     * @brief Check if local partition owns a physical boundary
     * @param direction 0=x, 1=y, 2=z
     * @param side LEFT or RIGHT
     * @return true if this partition contains the physical boundary
     */
    bool owns_physical_boundary(Dim direction, BoundarySide side) const {
        int idx = direction * 2 + static_cast<int>(side);
        return local_partition.has_boundary[idx];
    }

    /**
     * @brief Get the Cartesian communicator
     */
    MPI_Comm get_cart_comm() const { return comm_cart; }

    /**
     * @brief Get process grid dimensions
     */
    std::array<int, 3> get_process_grid() const { return {Px, Py, Pz}; }

    /**
     * @brief Get current rank
     */
    int get_rank() const { return rank; }

    /**
     * @brief Get total number of processes
     */
    int get_size() const { return size; }

    /**
     * @brief Get global dimensions
     */
    std::array<Dim, 3> get_global_dimensions() const {
        return {Nx_global, Ny_global, Nz_global};
    }

    /**
     * @brief Get local partition structure
     */
    const LocalPartition& get_local_partition() const {
        return local_partition;
    }

private:
    // Global grid dimensions
    Dim Nx_global, Ny_global, Nz_global;

    // MPI communicators
    MPI_Comm comm_world;
    MPI_Comm comm_cart;

    // Process information
    int rank, size;
    int Px, Py, Pz;  // Process grid dimensions
    std::array<int, 3> coords;  // Cartesian coordinates of this rank

    // Local partition information
    LocalPartition local_partition;

    /**
     * @brief Compute optimal process grid dimensions
     * Minimizes surface area (communication volume) for given number of processes
     */
    void compute_process_grid(int Px_user, int Py_user, int Pz_user) {
        // If user specified all dimensions, use them
        if (Px_user > 0 && Py_user > 0 && Pz_user > 0) {
            if (Px_user * Py_user * Pz_user != size) {
                throw std::runtime_error("DomainDecomposition: Px*Py*Pz must equal number of processes");
            }
            Px = Px_user;
            Py = Py_user;
            Pz = Pz_user;
            return;
        }

        // Automatic decomposition: minimize surface area
        // Surface area = 2*(Nx/Px * Ny/Py + Ny/Py * Nz/Pz + Nz/Pz * Nx/Px)
        // We want to find Px, Py, Pz such that Px*Py*Pz = size
        // and the aspect ratio matches the domain aspect ratio

        double aspect_xy = static_cast<double>(Nx_global) / Ny_global;
        double aspect_xz = static_cast<double>(Nx_global) / Nz_global;
        double aspect_yz = static_cast<double>(Ny_global) / Nz_global;

        Dim best_Px = 1, best_Py = 1, best_Pz = size;
        double min_surface = 1e30;

        // Try all factorizations of size
        for (int px = 1; px <= size; ++px) {
            if (size % px != 0) continue;
            int remaining = size / px;
            for (int py = 1; py <= remaining; ++py) {
                if (remaining % py != 0) continue;
                int pz = remaining / py;

                // Compute surface area metric
                double nx_local = static_cast<double>(Nx_global) / px;
                double ny_local = static_cast<double>(Ny_global) / py;
                double nz_local = static_cast<double>(Nz_global) / pz;
                double surface = 2.0 * (nx_local * ny_local + ny_local * nz_local + nz_local * nx_local);

                if (surface < min_surface) {
                    min_surface = surface;
                    best_Px = px;
                    best_Py = py;
                    best_Pz = pz;
                }
            }
        }

        Px = best_Px;
        Py = best_Py;
        Pz = best_Pz;
    }

    /**
     * @brief Create 3D Cartesian MPI communicator
     */
    void create_cartesian_grid() {
        int dims[3] = {Px, Py, Pz};
        int periods[3] = {0, 0, 0};  // Non-periodic boundaries
        int reorder = 1;  // Allow reordering for better performance

        MPI_Cart_create(comm_world, 3, dims, periods, reorder, &comm_cart);

        // Get this rank's coordinates in the Cartesian grid
        int coords_arr[3];
        MPI_Cart_coords(comm_cart, rank, 3, coords_arr);
        coords = {coords_arr[0], coords_arr[1], coords_arr[2]};
    }

    /**
     * @brief Setup local partition for this rank
     */
    void setup_local_partition() {
        // Compute local dimensions (distribute as evenly as possible)
        local_partition.Nx_local = Nx_global / Px + (coords[0] < Nx_global % Px ? 1 : 0);
        local_partition.Ny_local = Ny_global / Py + (coords[1] < Ny_global % Py ? 1 : 0);
        local_partition.Nz_local = Nz_global / Pz + (coords[2] < Nz_global % Pz ? 1 : 0);

        // Compute start indices
        local_partition.i_start = coords[0] * (Nx_global / Px) + std::min(coords[0], Nx_global % Px);
        local_partition.j_start = coords[1] * (Ny_global / Py) + std::min(coords[1], Ny_global % Py);
        local_partition.k_start = coords[2] * (Nz_global / Pz) + std::min(coords[2], Nz_global % Pz);

        // Compute end indices (exclusive)
        local_partition.i_end = local_partition.i_start + local_partition.Nx_local;
        local_partition.j_end = local_partition.j_start + local_partition.Ny_local;
        local_partition.k_end = local_partition.k_start + local_partition.Nz_local;

        // Get neighbor ranks using MPI_Cart_shift
        MPI_Cart_shift(comm_cart, 0, 1, &local_partition.neighbors[0], &local_partition.neighbors[1]); // x-direction
        MPI_Cart_shift(comm_cart, 1, 1, &local_partition.neighbors[2], &local_partition.neighbors[3]); // y-direction
        MPI_Cart_shift(comm_cart, 2, 1, &local_partition.neighbors[4], &local_partition.neighbors[5]); // z-direction

        // Determine physical boundary flags
        local_partition.has_boundary[0] = (coords[0] == 0);           // x- boundary
        local_partition.has_boundary[1] = (coords[0] == Px - 1);      // x+ boundary
        local_partition.has_boundary[2] = (coords[1] == 0);           // y- boundary
        local_partition.has_boundary[3] = (coords[1] == Py - 1);      // y+ boundary
        local_partition.has_boundary[4] = (coords[2] == 0);           // z- boundary
        local_partition.has_boundary[5] = (coords[2] == Pz - 1);      // z+ boundary
    }

    #else
    // Serial version (no MPI)
    DomainDecomposition(Dim Nx_global, Dim Ny_global, Dim Nz_global,
                        int Px_user = -1, int Py_user = -1, int Pz_user = -1)
        : Nx_global(Nx_global), Ny_global(Ny_global), Nz_global(Nz_global)
    {
        // In serial mode, everything is local
        local_partition.Nx_local = Nx_global;
        local_partition.Ny_local = Ny_global;
        local_partition.Nz_local = Nz_global;
        local_partition.i_start = 0;
        local_partition.j_start = 0;
        local_partition.k_start = 0;
        local_partition.i_end = Nx_global;
        local_partition.j_end = Ny_global;
        local_partition.k_end = Nz_global;
        
        // No neighbors in serial mode
        local_partition.neighbors.fill(-1);
        
        // All boundaries are physical boundaries
        local_partition.has_boundary.fill(true);
    }

    std::array<Dim, 3> get_local_dimensions() const {
        return {Nx_global, Ny_global, Nz_global};
    }

    std::array<Dim, 3> get_local_start_indices() const {
        return {0, 0, 0};
    }

    std::array<Dim, 3> get_local_end_indices() const {
        return {Nx_global, Ny_global, Nz_global};
    }

    bool global_to_local(Dim i_global, Dim j_global, Dim k_global,
                         Dim& i_local, Dim& j_local, Dim& k_local) const {
        i_local = i_global;
        j_local = j_global;
        k_local = k_global;
        return true;
    }

    void local_to_global(Dim i_local, Dim j_local, Dim k_local,
                         Dim& i_global, Dim& j_global, Dim& k_global) const {
        i_global = i_local;
        j_global = j_local;
        k_global = k_local;
    }

    bool is_interface_point(Dim i_local, Dim j_local, Dim k_local, Dim direction) const {
        return false;  // No interfaces in serial mode
    }

    std::pair<int, int> get_neighbors(Dim direction) const {
        return {-1, -1};  // No neighbors in serial mode
    }

    bool owns_physical_boundary(Dim direction, BoundarySide side) const {
        return true;  // All boundaries are physical in serial mode
    }

    int get_rank() const { return 0; }
    int get_size() const { return 1; }

    std::array<Dim, 3> get_global_dimensions() const {
        return {Nx_global, Ny_global, Nz_global};
    }

    const LocalPartition& get_local_partition() const {
        return local_partition;
    }

private:
    Dim Nx_global, Ny_global, Nz_global;
    LocalPartition local_partition;
    #endif
};

#endif // DOMAINDECOMPOSITION_HPP
