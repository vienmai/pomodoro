#ifndef OPTIMIZER_LBFGS_HPP_
#define OPTIMIZER_LBFGS_HPP_

#include "terminator.hpp"
#include "utility/null.hpp"
#include "stepsize.hpp"
#include "parameters.hpp"
#include <deque>

namespace optimizer {
template <class value_t = double, 
          template <class> class stepsize = stepsize::linesearch>
struct lbfgs : public stepsize<value_t> {
    using param_t = optimizer::lbfgs_params<value_t>;
    using vector_t = std::vector<value_t>;
    using blas_t = algebra::matrix::blas<value_t>;
    lbfgs() = default;
    lbfgs(param_t _params) : params(std::move(_params)) {}

    void initialize(const vector_t &x0) noexcept {
        k       = 0;
        m       = params.memory;
        int n   = x0.size();
        x       = x0;
        xtest   = vector_t(n);
        g       = vector_t(n);
        p       = vector_t(n);
        dx      = vector_t(n);
        dg      = vector_t(n);
        xprev   = vector_t(n);
        gprev   = vector_t(n);
        alpha   = vector_t(m);
        xb      = x.data();
        gb      = g.data();
        xb_c    = xb;
        xe_c    = xb + n;
        gb_c    = gb;
        s.resize(0);
        y.resize(0);
        rho.resize(0);
    }
    template <class... Ts> void stepsize_parameters(Ts &&... params) {
        stepsize<value_t>::parameters(std::forward<Ts>(params)...);
    }
    template <class Loss, class Logger, class Terminator>
    void solve(Loss &&loss, Logger &&logger, Terminator &&terminator) noexcept {
        if (params.verbose) 
            printf("%6s %10s\n", "Iter", "Fx");
        value_t stz = params.init_step;
        while (!terminator(k, fx, xb_c, xe_c, gb_c)) {
            if (k == 0) {
                fx = std::forward<Loss>(loss)(xb, gb);
                xprev = x;
                gprev = g;
            } else {
                p = g;
                for (int i = 0; i < s.size(); i++) {
                    alpha[i] = rho[i] * algebra::vdot(s[i], p);
                    blas_t::axpy(p.size(), -alpha[i], y[i].data(), 1, p.data(), 1);
                }
                value_t gamma = s.size() > 0 ? algebra::vdot(s[0], y[0]) / algebra::vdot(y[0], y[0]) : 1.0;
                std::transform(p.begin(), p.end(), p.begin(),
                               [gamma](const value_t &val) { return gamma * val; });
                for (int i = s.size() - 1; i > -1; i--) {
                    auto beta = rho[i] * algebra::vdot(y[i], p);
                    blas_t::axpy(p.size(), alpha[i] - beta, s[i].data(), 1, p.data(), 1);
                }
                stz = params.ls_adapt ? std::max(stz * params.ls_ratio, params.max_step) : params.init_step;
                auto t1 = algebra::vdot(g, p);
                for (int idx = 0; idx < params.ls_maxiter; idx++) {
                    std::transform(x.begin(), x.end(), p.begin(), xtest.begin(),
                                   [stz](const value_t &xval, const value_t &pval) 
                                   { return xval - stz * pval; });
                    if ((!params.is_ls) || (loss(xtest.data()) <= fx - params.ls_dec * stz * t1)) {
                        break;
                    }
                    stz *= params.ls_rho;
                }
                x = xtest;
                fx = std::forward<Loss>(loss)(xb, gb);

                if (algebra::ltwo(g) < params.epsilon) {
                    printf("lbfgs done!\n");
                    break;
                }
                if (k > m) {
                    s.pop_back();
                    y.pop_back();
                    rho.pop_back();
                }

                algebra::sub(x, xprev, dx);
                algebra::sub(g, gprev, dg);
                s.push_front(dx);
                y.push_front(dg);
                rho.push_front(1.0 / algebra::vdot(dx, dg));

                xprev = x;
                gprev = g;
            }
            if (k % params.log_every == 0)
                logger(k, fx, xb_c, xe_c);
            if ((k % params.print_every == 0) && (params.verbose))
                printf("%6d %10.5f\n", k, fx);
            k++;
        }
    }
    template <class Loss, class Logger>
    void solve(Loss &&loss, Logger &&logger) noexcept {
        solve(std::forward<Loss>(loss), logger,
              terminator::iteration<value_t>{params.max_iters});
    }
    template <class Loss>
    void solve(Loss &&loss) noexcept {
        solve(std::forward<Loss>(loss), utility::detail::null{},
              terminator::iteration<value_t>{params.max_iters});
    }
    vector_t getx() const noexcept { return x; }
    value_t getf() const noexcept { return fx; }

private:
    int k, m;
    value_t fx;
    vector_t x, xtest, g, p, xprev, gprev, dx, dg;
    value_t *xb, *xe, *gb;
    const value_t *xb_c, *xe_c, *gb_c;
    std::deque<vector_t> s, y;
    std::deque<value_t> rho;
    vector_t alpha;
    param_t params;
};
} // namespace optimizer
#endif