#ifndef PROX_L1NORM_HPP_
#define PROX_L1NORM_HPP_

#include <algorithm>
#include <cmath>

namespace prox {
template <class value_t> 
struct l1norm {
    l1norm(const value_t lambda = 1) : lambda_{lambda} {}

    l1norm(const l1norm &) = default;
    l1norm &operator=(const l1norm &) = default;
    l1norm(l1norm &&) = default;
    l1norm &operator=(l1norm &&) = default;

    std::vector<value_t> operator()(const std::vector<value_t> &x) const noexcept {
        value_t temp;
        auto &lambda = lambda_;
        std::vector<value_t> res(x.size());
        std::transform(std::begin(x), std::end(x), std::begin(res), 
                       [&](const value_t val) {
                           temp = std::max(std::abs(val) - lambda, value_t{0});
                           return val < value_t{0} ? - temp : temp;
                       });
        return res;
    }

    void proxgrad(const value_t step, const std::vector<value_t> &x, 
    const std::vector<value_t> &g, std::vector<value_t> &xout) const noexcept {
        value_t xgd, temp;
        auto &lambda = lambda_;
        std::transform(std::begin(x), std::end(x), std::begin(g), std::begin(xout),
                       [&step, &xgd, &temp, &lambda](const value_t xval, const value_t gval) {
                           xgd = xval - step * gval;
                           temp = std::max(std::abs(xgd) - step * lambda, value_t{0});
                           return xgd < value_t{0} ? -temp : temp;
                       });
    }
    
protected:
    void parameters(const value_t lambda) { lambda_ = lambda; }

private:
    value_t lambda_{1};
};
} // namespace prox

#endif
