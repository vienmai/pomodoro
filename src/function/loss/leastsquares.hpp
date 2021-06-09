#ifndef FUNCTION_LOSS_LEASTSQUARES_HPP_
#define FUNCTION_LOSS_LEASTSQUARES_HPP_

#include <iterator>
#include "aloss.hpp"

namespace function {
namespace loss {
template <class value_t>
struct leastsquares : public aloss<value_t> {
    leastsquares() = default;
    leastsquares(data<value_t> data)
        : aloss<value_t>(std::move(data)) {}

    value_t operator()(const value_t *x) const noexcept override {
        value_t loss{0};
        std::vector<value_t> residual(aloss<value_t>::nsamples());
        aloss<value_t>::data_.residual(x, residual.data());
        for (const value_t r : residual)
            loss += 0.5 * r * r;
        return loss;
    }    
    value_t operator()(const value_t *x, value_t *g) const noexcept override {
        value_t loss{0};
        std::vector<value_t> residual(aloss<value_t>::nsamples());
        aloss<value_t>::data_.residual(x, residual.data());
        for (const value_t r : residual)
            loss += 0.5 * r * r;
        aloss<value_t>::matrix()->mult_add('t', 1, residual.data(), 0, g);
        return loss;
    }
    value_t operator()(const value_t *x, value_t *g, const int *ib,
                      const int *ie) const noexcept override {
        value_t loss{0};
        std::vector<value_t> residual(std::distance(ib, ie));
        aloss<value_t>::data_.residual(x, residual.data(), ib, ie);
        for (const value_t r : residual)
            loss += 0.5 * r * r;
        aloss<value_t>::matrix()->mult_add('t', 1, residual.data(), 0, g, ib, ie);
        return loss;
    }
    
    void grad(const value_t *x, value_t *g) const noexcept override {
        std::vector<value_t> residual(aloss<value_t>::nsamples());
        aloss<value_t>::data_.residual(x, residual.data());
        aloss<value_t>::matrix()->mult_add('t', 1, residual.data(), 0, g);
    }
};

} // namespace loss
} // namespace function



#endif
