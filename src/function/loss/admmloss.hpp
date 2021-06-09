#ifndef FUNCTION_LOSS_ADMMLOSS_HPP_
#define FUNCTION_LOSS_ADMMLOSS_HPP_

#include <cmath>
#include <numeric>
#include "aloss.hpp"

namespace function {
namespace loss {
template <class value_t, template <class> class aloss>
struct admmloss : public aloss<value_t> {
    using loss_t = aloss<value_t>;
    using vector_t = std::vector<value_t>;
    using blas_t = algebra::matrix::blas<value_t>;
    admmloss() = default;
    admmloss(data<value_t> data) : loss_t(std::move(data)) {}

    void setinput(const vector_t &xglob, const vector_t &yloc,
                  const value_t rho, const value_t sign) {
        this->xglob   = xglob;
        this->yloc    = yloc;
        this->rho     = rho;
        this->sign    = sign;
        this->xdiff   = vector_t(xglob.size());
    }
    value_t getrho() const noexcept { return rho; }
    value_t getf(const vector_t &x) noexcept { 
        return loss_t::operator()(x.data()); 
    }
    value_t operator()(const value_t *x) noexcept {
        auto loss = loss_t::operator()(x);
        algebra::sub(x, xglob, xdiff);
        auto t1 = algebra::vdot(yloc, xdiff);
        auto t2 = algebra::ltwo(xdiff);
        loss += sign * t1 + 0.5 * rho * t2 * t2;
        return loss;
    }
    value_t operator()(const value_t *x, value_t *g) noexcept {
        auto loss = loss_t::operator()(x, g);
        algebra::sub(x, xglob, xdiff);
        auto t1 = algebra::vdot(yloc, xdiff);
        auto t2 = algebra::ltwo(xdiff);
        loss += sign * t1 + 0.5 * rho * t2 * t2;
        blas_t::axpy(yloc.size(), sign, &yloc[0], 1, g, 1);
        blas_t::axpy(xglob.size(), rho, &xdiff[0], 1, g, 1);
        return loss;
    }
    void grad(const value_t *x, value_t *g) noexcept {
        loss_t::grad(x, g);
        algebra::sub(x, xglob, xdiff);
        blas_t::axpy(yloc.size(), sign, &yloc[0], 1, g, 1);
        blas_t::axpy(xglob.size(), rho, &xdiff[0], 1, g, 1);
    }

private:
    value_t rho, sign; // sign to represent sign * <yloc, x - xglob>
    std::vector<value_t> yloc, xglob, xdiff;
};

template <class value_t = double>
using admm_logistic = loss::admmloss<value_t, loss::logistic_>;
template <class value_t = double>
using admm_leastsquare = loss::admmloss<value_t, loss::leastsquares>;
} // namespace loss
} // namespace function

#endif
