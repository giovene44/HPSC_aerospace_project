#include <iostream>
#include <cassert>
#include <cmath>
#include "DomainDecomposition.hpp"
#include "Variables.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

int main(int argc, char** argv) {
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        std::cout << "=== Domain Decomposition Test ===" << std::endl;
        std::cout << "Running on " << size << " process(es)" << std::endl;
    }
    
    // Test 1: Create decomposition with automatic partitioning
    Dim Nx = 32, Ny = 32, Nz = 32;
    DomainDecomposition decomp(Nx, Ny, Nz, MPI_COMM_WORLD, -1, -1, -1);
    
    // Test 2: Verify local dimensions
    auto local_dims = decomp.get_local_dimensions();
    Dim Nx_local = local_dims[0];
    Dim Ny_local = local_dims[1];
    Dim Nz_local = local_dims[2];
    
    auto start_indices = decomp.get_local_start_indices();
    Dim i_start = start_indices[0];
    Dim j_start = start_indices[1];
    Dim k_start = start_indices[2];
    
    auto end_indices = decomp.get_local_end_indices();
    Dim i_end = end_indices[0];
    Dim j_end = end_indices[1];
    Dim k_end = end_indices[2];
    
    std::cout << "Rank " << rank << ": Local dims = (" << Nx_local << ", " << Ny_local << ", " << Nz_local << ")" << std::endl;
    std::cout << "Rank " << rank << ": Start indices = (" << i_start << ", " << j_start << ", " << k_start << ")" << std::endl;
    std::cout << "Rank " << rank << ": End indices = (" << i_end << ", " << j_end << ", " << k_end << ")" << std::endl;
    
    // Test 3: Verify global coverage
    int local_count = Nx_local * Ny_local * Nz_local;
    int global_count = 0;
    MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        int expected_count = Nx * Ny * Nz;
        std::cout << "Global coverage check: " << global_count << " / " << expected_count << std::endl;
        assert(global_count == expected_count && "All grid points must be covered exactly once");
    }
    
    // Test 4: Verify neighbor identification
    auto proc_grid = decomp.get_process_grid();
    int Px = proc_grid[0];
    int Py = proc_grid[1];
    int Pz = proc_grid[2];
    if (rank == 0) {
        std::cout << "Process grid: " << Px << " x " << Py << " x " << Pz << std::endl;
    }
    
    auto neighbors_x = decomp.get_neighbors(0);
    int left_x = neighbors_x.first;
    int right_x = neighbors_x.second;
    
    auto neighbors_y = decomp.get_neighbors(1);
    int left_y = neighbors_y.first;
    int right_y = neighbors_y.second;
    
    auto neighbors_z = decomp.get_neighbors(2);
    int left_z = neighbors_z.first;
    int right_z = neighbors_z.second;
    
    std::cout << "Rank " << rank << ": Neighbors X=[" << left_x << ", " << right_x << "]"
              << " Y=[" << left_y << ", " << right_y << "]"
              << " Z=[" << left_z << ", " << right_z << "]" << std::endl;
    
    // Test 5: Verify boundary ownership
    bool has_left_x = decomp.owns_physical_boundary(0, BoundarySide::LEFT);
    bool has_right_x = decomp.owns_physical_boundary(0, BoundarySide::RIGHT);
    
    std::cout << "Rank " << rank << ": Owns X boundaries: [" << has_left_x << ", " << has_right_x << "]" << std::endl;
    
    // Test 6: Test global-to-local mapping
    for (Dim i = i_start; i < i_end; ++i) {
        for (Dim j = j_start; j < j_end; ++j) {
            for (Dim k = k_start; k < k_end; ++k) {
                Dim i_local, j_local, k_local;
                bool in_partition = decomp.global_to_local(i, j, k, i_local, j_local, k_local);
                assert(in_partition && "Point should be in local partition");
                
                // Verify inverse mapping
                Dim i_global, j_global, k_global;
                decomp.local_to_global(i_local, j_local, k_local, i_global, j_global, k_global);
                assert(i == i_global && j == j_global && k == k_global && "Inverse mapping failed");
            }
        }
    }
    
    if (rank == 0) {
        std::cout << "\n=== All tests passed! ===" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
#else
    std::cout << "This test requires MPI. Please compile with USE_MPI=1" << std::endl;
    return 1;
#endif
}
