#ifndef FUNCTION_LOSS_LOGISTIC__HPP_
#define FUNCTION_LOSS_LOGISTIC__HPP_

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include "aloss.hpp"

namespace function {
namespace loss {
template <class value_t>
struct logistic_ : public aloss<value_t> {
    logistic_() = default;
    logistic_(data<value_t> data) : aloss<value_t>(std::move(data)) {
        A = aloss<value_t>::matrix();
        auto b = aloss<value_t>::labels();
        A->scale_rows(b->data());
        bAx = std::vector<value_t>(b->size());
    }
    value_t operator()(const value_t *x) noexcept override {
        A->mult_add('n', -1.0, x, 0, &bAx[0]); 
        auto loss = std::accumulate(std::begin(bAx), std::end(bAx), 0.0,
                       [&](value_t loss, const value_t val) {
                           const value_t temp = std::exp(val);
                           return std::isinf(temp) ? loss + val : loss + std::log1p(temp);
                       });
        return loss;
    }
    value_t operator()(const value_t *x, value_t *g) noexcept override {
        value_t loss{0};
        A->mult_add('n', -1, x, 0, &bAx[0]); 
        std::transform(std::begin(bAx), std::end(bAx), std::begin(bAx),
                        [&](const value_t val) {
                        const value_t temp = std::exp(val);
                        if (std::isinf(temp)){
                            loss += val;
                            return - 1.0;
                        }else{
                            loss += std::log1p(temp);
                            return - temp / (1.0 + temp);
                        }
                        });
        A->mult_add('t', 1.0 , &bAx[0], 0, g);
        return loss;
    }
    value_t operator()(const value_t *x, value_t *g, const int *ib,
                      const int *ie) noexcept override {
        value_t loss{0};
        std::vector<value_t> ax(std::distance(ib, ie));//need improvement?
        A->mult_add('n', -1, x, 0, &bAx[0], ib, ie);
        for (auto &val : bAx) {
            const value_t temp = std::exp(val);
            if (std::isinf(temp)) {
                loss += val;
                val = -1;
            }else{
                loss += std::log1p(temp);
                val = - temp / (1 + temp);
            }
        }
        A->mult_add('t', 1, &bAx[0], 0, g, ib, ie);
        return loss;
    }

    void grad(const value_t *x, value_t *g) noexcept override {
        A->mult_add('n', -1, x, 0, &bAx[0]);
        std::transform(std::begin(bAx), std::end(bAx), std::begin(bAx),
                       [&](const value_t val) {
                           const value_t temp = std::exp(val);
                           return std::isinf(temp) ? -1 : - temp / (1 + temp);
                       });
        A->mult_add('t', 1.0, &bAx[0], 0, g);
    }
private:
    std::vector<value_t>  bAx;
    std::shared_ptr<matrix::amatrix<value_t>> A;
};
} // namespace loss
} // namespace function

#endif
