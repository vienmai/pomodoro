#include <iostream>
#include <algorithm>
#include "algebra.hpp"
#include "matrix.hpp"
#include "function.hpp"
#include "prox.hpp"
#include "terminator.hpp"
#include "optimizer.hpp"
#include "utility.hpp"

using namespace matrix;

int main(int argc, char *argv[]) {
    int m = 3, n = 5000;
    auto res = utility::randn<double>(n);
    auto resvec = utility::randn<double>(m * n);

    auto mem = m;
    std::vector<double> rtr(mem * mem);
    std::vector<double> rtr_(mem * mem);

    int idx = 1;
    auto offset = idx * n;

    utility::timer t;
    algebra::matrix::blas<double>::syr('u', mem, -1, resvec.data() + offset, 1, rtr.data(), mem);
    algebra::matrix::blas<double>::syr('u', mem, 1, res.data(), 1, rtr.data(), mem);
    std::copy(res.begin(), res.end(), resvec.begin() + offset);
    
    t.elapsed();
    std::copy(res.begin(), res.end(), resvec.begin() + offset);
    algebra::matrix::blas<double>::gemm('t', 'n', mem, mem, n, 1, resvec.data(),
                                        n, resvec.data(), n, 0, rtr_.data(), mem);
    t.elapsed();

    utility::printvec(rtr);
    utility::printvec(rtr_);
    // std::cout
    //     << "outdiff: " << algebra::ltwo(out1) << '\n';
    return 0;
}