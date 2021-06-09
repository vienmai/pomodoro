#ifndef ACCELERATOR_APPA_HPP_
#define ACCELERATOR_APPA_HPP_

#include <vector>
#include "algebra.hpp"

namespace accelerator {
template <class value_t> struct appa {
    using blas_t = algebra::matrix::blas<value_t>;
    appa() = default;

    template <class Comm, class InputIt1, class InputIt2, class OutputIt1, class OutputIt2>
    void accelerate(const Comm &COMM, const int rank, const int k,
                    InputIt1 xbegin, InputIt1 xend, InputIt2 ybegin,
                    OutputIt1 ubegin, OutputIt2 vbegin) noexcept {
        if (k == 0){
            std::copy(xbegin, xend, xcur.begin()); // xcur <- x_{k+1}
            std::copy(ybegin, ybegin + n, ycur.begin());
        } else {
            value_t alpha = k / value_t (k + 2);
            std::transform(xcur.begin(), xcur.end(), uprev.begin(), uprev.begin(),
                           [alpha](const value_t &xval, const value_t &uval) 
                           { return alpha * (-2 * xval + uval); });
            std::transform(ycur.begin(), ycur.end(), vprev.begin(), vprev.begin(),
                           [alpha](const value_t &yval, const value_t &vval) 
                           { return alpha * (-2 * yval + vval); });
            std::copy(xbegin, xend, xcur.begin()); // xcur <- x_{k+1}
            std::copy(ybegin, ybegin + n, ycur.begin()); // This order is critial, the next line will change 
                                                         // (xbegin, xend) since xbegin == ubegin 
            std::copy(uprev.begin(), uprev.end(), ubegin); // u <- alpha * (-2 x_k + u_{k-1})
            std::copy(vprev.begin(), vprev.end(), vbegin); 
            blas_t::axpy(n, 1.0 + alpha, xcur.data(), 1, ubegin, 1); // u <- (1 + alpha) * x_{k+1} + u
            blas_t::axpy(n, 1.0 + alpha, ycur.data(), 1, vbegin, 1);
            uprev = ucur; // store u_k
            vprev = vcur;
        }
        std::copy(ubegin, ubegin + n, ucur.begin()); // ucur <- u_{k+1}
        std::copy(vbegin, vbegin + n, vcur.begin()); 
    }

protected:
    void parameter(){}
    void initialize(const std::vector<value_t> &x0) {
        xcur  = x0;
        ucur  = x0;
        uprev = x0;
        n     = x0.size();
        ycur  = std::vector<value_t>(n);
        vcur  = std::vector<value_t>(n);
        vprev = std::vector<value_t>(n);
    }

private:
    int n;
    std::vector<value_t> xcur, ycur, uprev, vprev, ucur, vcur;
};
} // namespace prox

#endif


