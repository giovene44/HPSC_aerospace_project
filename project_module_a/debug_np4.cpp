// Quick test to see what's happening with 4 processes
#include <iostream>
#include "SchurMatrixData.hpp"

int main() {
    int N = 100;
    int num_procs = 4;
    
    std::cout << "Domain decomposition for N=" << N << ", np=" << num_procs << ":\n";
    
    for (int rank = 0; rank < num_procs; ++rank) {
        SchurComplementData data;
        data.init(N, num_procs, rank);
        
        std::cout << "Rank " << rank << ": "
                  << "local_N=" << data.local_N 
                  << ", global_start=" << data.global_start
                  << ", n_internal=" << data.n_internal
                  << ", has_left=" << data.has_left_interface
                  << ", has_right=" << data.has_right_interface
                  << ", owns global[" << data.global_start 
                  << ".." << (data.global_start + data.local_N - 1) << "]"
                  << std::endl;
    }
    
    return 0;
}
