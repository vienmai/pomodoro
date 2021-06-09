#ifndef ACCELERATOR_ANDERSON_HPP_
#define ACCELERATOR_ANDERSON_HPP_

#include <vector>
#include "algebra.hpp"
#include "utility/mpitype.hpp"

namespace accelerator {
template <class value_t> struct anderson {
    using vector_t = std::vector<value_t>;
    using blas_t = algebra::matrix::blas<value_t>;
    using lapack_t = algebra::matrix::lapack<value_t>;
    anderson() = default;

    template <class Comm, class InputIt1, class InputIt2, class OutputIt1, class OutputIt2>
    void accelerate(const Comm &COMM, const int rank, const int k,
                    InputIt1 xbegin, InputIt1 xend, InputIt2 ybegin,
                    OutputIt1 ubegin, OutputIt2 vbegin) noexcept {
        algebra::sub(xbegin, res, res); // res <- x - u - v
        algebra::add(ybegin, res, res);  // r0 = res = x + y - u -v
        if (k == 0) {
            gxvec.insert(gxvec.begin(), xbegin, xend);       // g(x0)
            gyvec.insert(gyvec.begin(), ybegin, ybegin + n); // g(x0)
            resvec.insert(resvec.begin(), res.begin(), res.end()); // r0
        } else {
            auto mk = std::min(m, k);
            if (k < m + 1) {
                gxvec.insert(gxvec.end(), xbegin, xend);
                gyvec.insert(gyvec.begin(), ybegin, ybegin + n); // g(xk)
                resvec.insert(resvec.end(), res.begin(), res.end());
            } else {
                auto idx = m > 0 ? (k - 1) % m : 0;
                auto offset = idx * n;
                std::copy(xbegin, xend, gxvec.begin() + offset);
                std::copy(ybegin, ybegin + n, gyvec.begin() + offset);
                std::copy(res.begin(), res.end(), resvec.begin() + offset);
            }
            assert(gxvec.size() == (mk + 1) * n);
            assert(gyvec.size() == (mk + 1) * n);

            auto mem = mk + 1;
            std::vector<value_t> rtr_send(mem * mem);
            std::vector<value_t> rtr_recv(mem * mem);
            blas_t::gemm('t', 'n', mem, mem, n, 1, resvec.data(),
                         n, resvec.data(), n, 0, rtr_send.data(), mem);
            MPI_Allreduce(rtr_send.data(), rtr_recv.data(), rtr_send.size(),
                          utility::MPI_Type<value_t>(), MPI_SUM, COMM);
            auto normRR = algebra::ltwo(rtr_recv);
            std::transform(rtr_recv.begin(), rtr_recv.end(), rtr_recv.begin(),
                           [normRR](const value_t &val) { return val / normRR; });
            for (int i = 0; i < mem; i++) {
                rtr_recv[i * mem + i] += reg;
            }
            aa_linear_solve(rtr_recv, gxvec, gyvec, mem, reg, ubegin, vbegin);
        }
        algebra::add(ubegin, n, vbegin, res.data()); // res = u + v
    }

protected:
    void parameters(const value_t memory, const value_t reg) {
        this->m   = memory;
        this->reg = reg;
    }
    void initialize(const std::vector<value_t> &x0) {
        n    = x0.size();
        res  = x0;  // store uk + vk; v0 = 0
        gxvec.reserve(n * (m + 1));
        gyvec.reserve(n * (m + 1));
        resvec.reserve(n * (m + 1));
    }

private:
    int m{5}, n;
    value_t reg{1E-10};
    std::vector<value_t> res, gxvec, gyvec, resvec;

    void aa_linear_solve(vector_t &rtr, const vector_t &gxvec, const vector_t &gyvec,
                         const int mem, const value_t reg, value_t *out1, value_t *out2) {   
        std::vector<value_t> coeff(mem, 1.0);
        std::vector<int> ipiv(mem);
        auto failed = lapack_t::gesv(mem, 1, rtr.data(), mem, ipiv.data(), coeff.data(), mem);
        if (!failed) {
            auto sum = std::accumulate(coeff.begin(), coeff.end(), 0.0); // 0.0 -> double
            std::transform(coeff.begin(), coeff.end(), coeff.begin(),
                           [sum](const value_t &val) { return val / sum; });
            blas_t::gemv('n', n, mem, 1, gxvec.data(), n, coeff.data(), 1, 0, out1, 1);
            blas_t::gemv('n', n, mem, 1, gyvec.data(), n, coeff.data(), 1, 0, out2, 1);
        } else {
            std::cout << "Linear solver failed.\n";
            abort();
        }
    }
};
} // namespace accelerator

#endif


