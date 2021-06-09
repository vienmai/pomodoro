#include <iostream>
#include <cstdlib>
#include <algorithm>
#include "algebra.hpp"
#include "matrix.hpp"
#include "function.hpp"
#include "prox.hpp"
#include "optimizer.hpp"
#include "utility.hpp"
#include <chrono>

double randn_double() { return (rand() / (double)(RAND_MAX)) * 2 - 1; }
using blas_t = algebra::matrix::blas<double>;

int main(int argc, char *argv[]) {
    srand(42);
    int nrows = 200;
    int ncols = 200;
    std::vector<double> x(ncols);
    std::vector<double> y(nrows);
    std::vector<double> A(nrows*ncols);
    std::generate(std::begin(x), std::end(x), randn_double);
    std::generate(std::begin(y), std::end(y), randn_double);
    std::generate(std::begin(A), std::end(A), randn_double);

    int nruns = 100;
    std::chrono::time_point<std::chrono::high_resolution_clock> tstart{
        std::chrono::high_resolution_clock::now()}, tend;
    // double bnorm, bdot;
    for(int i = 0; i < nruns; i++){
        // bnorm = blas_t::nrm2(x.size(), x.data(), 1);
        // bdot = blas_t::dot(x.size(), x.data(), 1, y.data(), 1);
        blas_t::gemv('n', nrows, ncols, 1, A.data(), nrows, x.data(), 1, 0, y.data(), 1);
    }
    tend = std::chrono::high_resolution_clock::now();
    auto telapsed =
        std::chrono::duration<double, std::chrono::milliseconds::period>(tend - tstart);
    std::cout << "Elapsed time: " << telapsed.count() / (double)nruns << " ms\n";

    // std::cout << "xnorm: " << bnorm << std::endl;
    // std::cout << "dot: " << bdot << std::endl;
    return 0;
}

// utility::printvec(A);
// std::chrono::time_point<std::chrono::high_resolution_clock> tstart{
//     std::chrono::high_resolution_clock::now()},
//     tend;
// int nruns = 100;
// double vnorm, vdot;
// for (int i = 0; i < nruns; i++){
//     // vnorm = algebra::ltwo(x);
//     // vdot = algebra::vdot(x, y);
// }
// tend = std::chrono::high_resolution_clock::now();
// auto telapsed =
//     std::chrono::duration<double, std::chrono::milliseconds::period>(
//         tend - tstart);
// std::cout << telapsed.count() / (double) nruns << std::endl;

// std::cout << "xnorm: "<< vnorm << std::endl;
// std::cout << "dot: "<< vdot << std::endl;