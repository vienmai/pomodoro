#ifndef STEPSIZE_POLYAK_HPP_
#define STEPSIZE_POLYAK_HPP_

#include <numeric>

namespace stepsize {
template <class value_t> struct polyak : public astepsize<value_t> {
    polyak() = default;

    template <class Loss, class Prox, class InputIt1, class InputIt2>
    value_t get_stepsize(Loss &&loss, Prox &&prox, const int k, const value_t fx,
                         InputIt1 xbegin, InputIt1 xend, InputIt2 gcurr){
        auto n = std::distance(xbegin, xend);
        auto normsq = std::inner_product(gcurr, gcurr + n, gcurr, 0.0);
        return (fx - fmin) / normsq;
    }

protected:
    void parameters(const value_t fmin) { this->fmin = fmin; }
private:
    value_t fmin{0};
};
} // namespace stepsize

#endif
