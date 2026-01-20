#include <run_err.hpp>

int main(int argc, char **argv)
{
#ifdef USE_MPI
    return run_multiple_mpi(argc, argv);
#else
    (void)argc;
    (void)argv;
    bool use_openMP = false;
    return run_multiple(use_openMP);
#endif
}
