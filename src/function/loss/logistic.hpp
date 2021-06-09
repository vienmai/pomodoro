#ifndef FUNCTION_LOSS_LOGISTIC_HPP_
#define FUNCTION_LOSS_LOGISTIC_HPP_

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include "aloss.hpp"

namespace function {
namespace loss {
template <class value_t>
struct logistic : public aloss<value_t> {
    logistic() = default;
    logistic(data<value_t> data)
        : aloss<value_t>(std::move(data)) {}

    value_t operator()(const value_t *x) noexcept {
        std::vector<value_t> ax(aloss<value_t>::nsamples());
        auto A = aloss<value_t>::matrix();
        auto b = aloss<value_t>::labels();
        A->mult_add('n', 1, x, 0, &ax[0]);
        std::transform(std::begin(*b), std::end(*b), std::begin(ax), std::begin(ax),
                       [](const value_t b, const value_t ax) { return -b * ax; });
        auto loss = std::accumulate(std::begin(ax), std::end(ax), 0.0,
                       [&](value_t loss, const value_t val) {
                           const value_t temp = std::exp(val);
                           return std::isinf(temp) ? loss + val : loss + std::log1p(temp);
                       });
        return loss / b->size();
    }
    value_t operator()(const value_t *x, value_t *g) noexcept {
        value_t loss{0};
        std::vector<value_t> ax(aloss<value_t>::nsamples());
        auto A = aloss<value_t>::matrix();
        auto b = aloss<value_t>::labels();
        A->mult_add('n', 1, x, 0, &ax[0]);
        std::transform(std::begin(*b), std::end(*b), std::begin(ax), std::begin(ax),
                        [](const value_t b, const value_t ax) { return -b * ax; });
        std::transform(std::begin(*b), std::end(*b), std::begin(ax), std::begin(ax),
                        [&](const value_t b, const value_t val) {
                        const value_t temp = std::exp(val);
                        if (std::isinf(temp)){
                            loss += val;
                            return -b;
                        }else{
                            loss += std::log1p(temp);
                            return -b * temp / (1 + temp);
                        }
                        });
        A->mult_add('t', 1, &ax[0], 0, g);
        return loss / b->size();
    }
    value_t operator()(const value_t *x, value_t *g, const int *ib,
                      const int *ie) noexcept {
        value_t loss{0};
        std::vector<value_t> ax(std::distance(ib, ie));
        auto A = aloss<value_t>::matrix();
        auto b = aloss<value_t>::labels();
        A->mult_add('n', 1, x, 0, &ax[0], ib, ie);
        const int *itemp{ib};
        int idx = 0;
        while (itemp != ie)
            ax[idx++] *= -(*b)[*itemp++];
        itemp = ib;
        for (auto &val : ax) {
            const value_t temp = std::exp(val);
            if (std::isinf(temp)) {
                loss += val;
                val = -(*b)[*itemp++];
            }else{
                loss += std::log1p(temp);
                val = -(*b)[*itemp++] * temp / (1 + temp);
            }
        }
        A->mult_add('t', 1, &ax[0], 0, g, ib, ie);
        return loss;
    }

    void grad(const value_t *x, value_t *g) noexcept {
        std::vector<value_t> ax(aloss<value_t>::nsamples());
        auto A = aloss<value_t>::matrix();
        auto b = aloss<value_t>::labels();
        A->mult_add('n', 1, x, 0, &ax[0]);
        std::transform(std::begin(*b), std::end(*b), std::begin(ax), std::begin(ax),
                       [](const value_t b, const value_t ax) { return -b * ax; });
        std::transform(std::begin(*b), std::end(*b), std::begin(ax), std::begin(ax),
                       [&](const value_t b, const value_t val) {
                           const value_t temp = std::exp(val);
                           return std::isinf(temp) ? -b : -b * temp / (1 + temp);                           
                       });
        A->mult_add('t', 1, &ax[0], 0, g);
    }
};
} // namespace loss
} // namespace function

#endif
