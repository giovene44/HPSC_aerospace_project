#include <run_err.hpp>
#include <iostream>

#ifdef USE_MPI
#include <mpi.h>
#endif

int main(int argc, char** argv)
{
#ifdef USE_MPI
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        std::cout << "Running with MPI on " << size << " process(es)" << std::endl;
    }
    
    int result = run_multiple();
    
    // Finalize MPI
    MPI_Finalize();
    
    return result;
#else
    return run_multiple();
#endif
}