#ifndef OPTIMIZER_ANOPTIMIZER_HPP_
#define OPTIMIZER_ANOPTIMIZER_HPP_

#include <vector>
#include "parameters.hpp"

namespace optimizer {
template <class value_t> struct anoptimizer {
    using loss_t = function::loss::aloss<value_t>;
    anoptimizer() = default;
    anoptimizer(const loss_t &loss) : loss_(loss) {}

    loss_t loss() const noexcept { return loss_; }

    value_t fval(const value_t *x) const noexcept { return loss_(x); }
    value_t fval_grad(const value_t *x, value_t *g) const noexcept { return loss_(x, g); }
    void grad(const value_t *x, value_t *g) const noexcept { return loss_.grad(x, g); }

    virtual std::vector<value_t> initialize(const std::vector<value_t> &x0) noexcept = 0;
    virtual void step() noexcept = 0;
    virtual void solve() noexcept = 0;

    virtual value_t getf() const noexcept = 0;
    virtual std::vector<value_t> getx() const noexcept = 0;

    virtual ~anoptimizer() = default;

protected:
    const loss_t &loss_;
};
} // namespace optimizer
#endif



   