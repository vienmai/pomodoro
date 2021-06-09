#ifndef STEPSIZE_BB_HPP_
#define STEPSIZE_BB_HPP_

#include <vector>
#include "../algebra/helper.hpp"

namespace stepsize {
template <class value_t> struct BB : public astepsize<value_t> {
    BB() = default;

    template <class Loss, class Prox, class InputIt1, class InputIt2>
    value_t get_stepsize(Loss &&loss, Prox &&prox, const int k, const value_t fx,
                       InputIt1 xbegin, InputIt1 xend, InputIt2 gcurr) {
        if (k==0) {
            xprev = std::vector<value_t>(xbegin, xend);
            auto n = std::distance(xbegin, xend);
            gprev = std::vector<value_t>(gcurr, gcurr + n);
            s = std::vector<value_t>(n);
            y = std::vector<value_t>(n);
            return init_step_;
        }
        algebra::sub(xbegin, xprev, s);
        algebra::sub(gcurr, gprev, y);
        auto norm_s = algebra::ltwo(s);
        xprev = std::vector<value_t>(xbegin, xend);
        gprev = std::vector<value_t>(gcurr, gcurr + std::distance(xbegin, xend));
        return norm_s * norm_s / algebra::vdot(s, y);
    }

protected:
    void parameters(const value_t init_step) { init_step_ = init_step; }
    
    template <class InputIt1, class InputIt2>
    void initialize(InputIt1 xbegin, InputIt1 xend) {
        auto n = std::distance(xbegin, xend);
        xprev = std::vector<value_t>(n);
        gprev = std::vector<value_t>(n);
    }

private:
    value_t init_step_;
    std::vector<value_t> xprev, gprev, s, y;
};
} // namespace stepsize

#endif
