#ifndef ACCELERATOR_NESTEROV_HPP_
#define ACCELERATOR_NESTEROV_HPP_

#include <vector>
#include <cmath>
#include "algebra.hpp"

namespace accelerator {
template <class value_t> struct nesterov {
    nesterov() = default;

    template <class InputIt1, class InputIt2, class OutputIt>
    OutputIt accelerate(const int k, const value_t fval, InputIt1 xbegin,
                     InputIt1 xend, InputIt2 gbegin, OutputIt obegin) noexcept {
        if (k == 0){
            tprev = 0.5 * (1 + std::sqrt(5));
            xprev = std::vector<value_t>(xbegin, xend);
            return obegin;
        }
        tcurr = 0.5 * (1 + std::sqrt(1 + 4 * tprev * tprev));
        auto alpha = (tprev - 1) / tcurr;
        value_t temp;
        int idx = 0;
        xprev_b = xprev.data();
        while (xbegin != xend){
            temp = *xbegin++;
            *obegin++ = (1 + alpha) * temp - alpha * (*xprev_b++);
            xprev[idx] = temp; // xbegin overwrited by obegin
            idx++;
        }
        tprev = tcurr;
        return obegin;
    }
protected:
    void parameter(){}
    void initialize(){}

private:
    value_t tprev, tcurr;
    value_t *xprev_b;
    std::vector<value_t> xprev;
};
} // namespace prox

#endif


