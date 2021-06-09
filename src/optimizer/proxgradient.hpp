#ifndef OPTIMIZER_PROXGRADIENT_HPP_
#define OPTIMIZER_PROXGRADIENT_HPP_

#include "terminator.hpp"
#include "utility/null.hpp"
#include "stepsize.hpp"
#include "accelerator.hpp"
#include "prox.hpp"
#include "parameters.hpp"

namespace optimizer {
template <class value_t = double, 
          template <class> class stepsize = stepsize::constant,
          template <class> class accelerator = accelerator::none,
          template <class> class prox = prox::none>
struct proxgradient : public accelerator<value_t>,
                      public prox<value_t>,
                      public stepsize<value_t> {
    using param_t = optimizer::params<value_t>;
    using vector_t = std::vector<value_t>;
    proxgradient() = default;
    proxgradient(param_t params) : params_(std::move(params)) {}

    void initialize(const vector_t &x0) noexcept {
        k = 0;
        x = x0;
        int n = x.size();
        g = vector_t(n);
        xb = x.data();
        gb = g.data();
        xb_c = xb;
        xe_c = xb + n;
        gb_c = gb;
    }
    template <class... Ts> void stepsize_parameters(Ts &&... params) {
        stepsize<value_t>::parameters(std::forward<Ts>(params)...);
    }
    template <class... Ts> void accelerator_parameters(Ts &&... params) {
        accelerator<value_t>::parameters(std::forward<Ts>(params)...);
    }
    template <class Loss, class Logger, class Terminator>
    void solve(Loss &&loss, Logger &&logger, Terminator &&terminator) noexcept {
        if (params_.verbose) 
            printf("%6s %10s\n", "Iter", "Fx");
        while (!terminator(k, fx, xb_c, xe_c, gb_c)) {
            step(std::forward<Loss>(loss));
            if(k % params_.log_every == 0)
                logger(k, fx, xb_c, xe_c);
            if ((k % params_.print_every == 0) && (params_.verbose))
                printf("%6d %10.5f\n", k, fx);
            k++;
        }
    }
    template <class Loss, class Logger>
    void solve(Loss &&loss, Logger &&logger) noexcept {
        solve(std::forward<Loss>(loss), std::forward<Logger>(logger),
              terminator::iteration<value_t>{params_.max_iters});
    }
    template <class Loss>
    void solve(Loss &&loss) noexcept {
        solve(std::forward<Loss>(loss), utility::detail::null{},
              terminator::iteration<value_t>{params_.max_iters});
    }
    vector_t getx() const noexcept { return x; }
    value_t getf() const noexcept { return fx; }

private:
    int k;
    value_t fx;
    vector_t x, g;
    value_t *xb, *xe, *gb;
    const value_t *xb_c, *xe_c, *gb_c;
    param_t params_;

    template <class Loss>
    void step(Loss &&loss) noexcept {
        fx = loss(xb, gb);
        auto stz = this->get_stepsize(std::forward<Loss>(loss), this,
                                      k, fx, xb_c, xe_c, gb_c);  
        this->proxgrad(stz, xb_c, xe_c, gb_c, xb);
        this->accelerate(k, fx, xb_c, xe_c, gb_c, xb);
    }
};

template <class value_t = double, template <class> class stepsize = stepsize::constant>
using gradient_descent = optimizer::proxgradient<value_t, stepsize, accelerator::none, prox::none>;
template <class value_t = double, template <class> class stepsize = stepsize::constant>
using nesterov = optimizer::proxgradient<value_t, stepsize, accelerator::nesterov, prox::none>;
} // namespace optimizer
#endif