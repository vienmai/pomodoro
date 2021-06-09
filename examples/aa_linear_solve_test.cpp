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
template <class value_t>
void aa_linear_solve_1(const std::vector<value_t> &resvec,
                       const std::vector<value_t> &gxvec,
                       const int mem,
                       const value_t reg,
                       value_t *out);
template <class value_t>
void aa_linear_solve_2(const std::vector<value_t> &resvec,
                       const std::vector<value_t> &gxvec,
                       const int mem,
                       const value_t reg,
                       value_t *out);

int main(int argc, char *argv[]) {
    int m = 25, n = 5000;
    double reg = 0.0;
    auto res = utility::randn<double>(m * n);
    auto gx = utility::randn<double>(m * n);

    std::vector<double> out1(n);
    std::vector<double> out2(n);

    utility::timer t;
    aa_linear_solve_1(res, gx, m, reg, out1.data());
    t.elapsed();
    aa_linear_solve_2(res, gx, m, reg, out2.data());
    t.elapsed();

    algebra::sub(out1, out2, out1);
    std::cout << "outdiff: " << algebra::ltwo(out1) << '\n';
    return 0;
}
template <class value_t>
void aa_linear_solve_1(const std::vector<value_t> &resvec,
                     const std::vector<value_t> &gxvec,
                     const int mem,
                     const value_t reg,
                     value_t *out) {
    auto n = gxvec.size() / mem;
    std::vector<value_t> rtr(mem * mem);

    algebra::matrix::blas<value_t>::gemm('t', 'n', mem, mem, n, 1, resvec.data(),
                 n, resvec.data(), n, 0, rtr.data(), mem);
    auto normRR = algebra::ltwo(rtr); // upperbound for ||.||_2
    std::transform(rtr.begin(), rtr.end(), rtr.begin(),
                   [normRR](const value_t &val) { return val / normRR; });
    for (int i = 0; i < mem; i++) {
        rtr[i * mem + i] += reg;
    }
    std::vector<value_t> coeff(mem, 1.0);
    std::vector<int> ipiv(mem);
    auto fail = algebra::matrix::lapack<value_t>::gesv(mem, 1, rtr.data(), mem, ipiv.data(), coeff.data(), mem);
    if (!fail) {
        auto sum = std::accumulate(coeff.begin(), coeff.end(), 0);
        std::transform(coeff.begin(), coeff.end(), coeff.begin(),
                       [sum](const value_t &val) { return val / sum; });
        algebra::matrix::blas<value_t>::gemv('n', n, mem, 1, gxvec.data(), n, coeff.data(), 1, 0, out, 1);
    }
    else {
        std::cout << "Linear solver failed.\n";
        abort();
    }
}
template <class value_t>
void aa_linear_solve_2(const std::vector<value_t> &resvec,
                       const std::vector<value_t> &gxvec,
                       const int mem,
                       const value_t reg,
                       value_t *out)
{
    auto n = gxvec.size() / mem;
    matrix::dmatrix<value_t> R(n, mem, resvec); // unnecessary copy
    matrix::dmatrix<value_t> G(n, mem, gxvec); // call blas directly on resvec
    std::vector<value_t> rtr(mem * mem);

    R.matrix_mult_add(1, 0, rtr.data()); // RR = R.T.R
    auto normRR = algebra::ltwo(rtr); // upperbound for ||.||_2
    std::transform(rtr.begin(), rtr.end(), rtr.begin(),
                   [normRR](const value_t &val) { return val / normRR; });
    for (int i = 0; i < mem; i++) {
        rtr[i * mem + i] += reg;
    }
    std::vector<value_t> coeff(mem, 1.0);
    std::vector<int> ipiv(mem);
    auto fail = algebra::matrix::lapack<value_t>::gesv(mem, 1, rtr.data(), mem, ipiv.data(), coeff.data(), mem);
    if (!fail) {
        auto sum = std::accumulate(coeff.begin(), coeff.end(), 0);
        std::transform(coeff.begin(), coeff.end(), coeff.begin(),
                       [sum](const value_t &val) { return val / sum; });
        G.mult_add('n', 1, coeff.data(), 0, out);
    }
    else {
        std::cout << "Linear solver failed.\n";
        abort();
    }
}