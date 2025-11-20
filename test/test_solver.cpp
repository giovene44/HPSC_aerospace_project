// #include <cassert>
// #include <cmath>
// #include <iostream>
// #include <stdexcept>
// #include <chrono>
// #include <vector>

// #include "navier_stokes_brinkman.hpp"

// void test_solver_pressure()
// {

//     const Dim Nx = 4, Ny = 5, Nz = 6;
//     const float dt = 0.01f;
//     const Real dx = 0.1f, dy = 0.1f, dz = 0.1f;

//     NavierStokesBrinkmann nsb(Nx, Ny, Nz, dt, 10.0f, dx, dy, dz);

//     // External forcing term: constant field f = (1, 2, 3)
//     for (int cmp = 0; cmp < 3; ++cmp)
//     {
//         for (Dim k = 0; k < Nz; ++k)
//             for (Dim j = 0; j < Ny; ++j)
//                 for (Dim i = 0; i < Nx; ++i)
//                 {
//                     nsb.f.set(cmp, i, j, k) = 1.0;
//                     nsb.k_field.set(i, j, k) = 1.0;
//                 }
//     }
//     nsb.initialize_gamma_field();
//     nsb.solve();
// }

// int main()
// {
//     try
//     {

//         test_solver_pressure();
//     }
//     catch (const std::exception &e)
//     {
//         std::cerr << "❌ Exception: " << e.what() << std::endl;
//         return 1;
//     }
//     catch (...)
//     {
//         std::cerr << "❌ Unknown error occurred." << std::endl;
//         return 1;
//     }

//     return 0;
// }

int main()
{   
    return 0;
}