#ifndef STEPSIZE_LINESEARCH_HPP_
#define STEPSIZE_LINESEARCH_HPP_

#include <vector>
#include "algebra.hpp"

namespace stepsize {
template <class value_t> 
struct linesearch {
    linesearch() = default;
    static const bool is_linesearch() noexcept { return true; }

    template <class Loss, class Prox, class InputIt1, class InputIt2>
    value_t get_stepsize(Loss &&loss, Prox &&prox, const int k, const value_t fx,
                         InputIt1 xbegin, InputIt1 xend, InputIt2 gcurr) noexcept {
        if (k==0) {
            auto n = std::distance(xbegin, xend);
            y = std::vector<value_t>(n);
            xdiff = std::vector<value_t>(n);
            ybegin = y.data();
        }
        prox->proxgrad(stepsize, xbegin, xend, gcurr, ybegin);
        auto fy = loss(ybegin);
        algebra::sub(xbegin, y, xdiff);
        auto t = algebra::ltwo(xdiff);
        while (fy > fx - algebra::vdot(gcurr, xdiff) + 0.5 * t * t / stepsize) {
            stepsize /= ls_ratio;
            prox->proxgrad(stepsize, xbegin, xend, gcurr, ybegin);
            fy = loss(ybegin);
            algebra::sub(xbegin, y, xdiff);
            t = algebra::ltwo(xdiff);
        }
        auto res = stepsize; 
        stepsize = ls_adapt ? std::max(stepsize * ls_ratio, max_step) : stepsize;
        return res;
    }

protected:
    void parameters(const value_t init_step, const value_t max_step,
                    const bool ls_adapt, const value_t ls_ratio) {
        this->stepsize = init_step;
        this->max_step = max_step;
        this->ls_adapt = ls_adapt;
        this->ls_ratio = ls_ratio;
    }

private:
    value_t *ybegin;
    std::vector<value_t> y, xdiff;
    bool ls_adapt{true};
    value_t stepsize, max_step{10}, ls_ratio{2};
};
} // namespace stepsize

#endif
