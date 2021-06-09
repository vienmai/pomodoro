#ifndef OPTIMIZER_AAPROX_HPP_
#define OPTIMIZER_AAPROX_HPP_

#include "terminator.hpp"
#include "utility/null.hpp"
#include "prox.hpp"
#include "parameters.hpp"
#include "matrix.hpp"
#include "stepsize.hpp"

namespace optimizer {
template <class value_t = double, 
          template <class> class stepsize = stepsize::constant,
          template <class> class prox = prox::none>
struct aaprox : public prox<value_t>, 
                public stepsize<value_t> {
    using param_t = optimizer::aa_params<value_t>;
    using vector_t = std::vector<value_t>;
    using blas_t = algebra::matrix::blas<value_t>;
    using lapack_t = algebra::matrix::lapack<value_t>;
    aaprox() = default;
    aaprox(param_t params) : params(std::move(params)) {}

    void initialize(const vector_t &x0) noexcept {
        k       = 0;
        m_      = params.memory;
        int n   = x0.size();
        x       = x0;
        y       = x0;
        g       = vector_t(n);
        gx      = vector_t(n);
        res     = vector_t(n);
        xprox   = vector_t(n);
        xb      = x.data();
        xe      = xb + n;
        gb      = g.data();
        xb_c    = xb;
        xe_c    = xb + n;
        gb_c    = gb;
        gxvec.reserve(n * (m_ + 1));
        resvec.reserve(n * (m_ + 1));
    }
    template <class... Ts> void stepsize_parameters(Ts &&... params) {
        stepsize<value_t>::parameters(std::forward<Ts>(params)...);
    }
    template <class... Ts> void prox_parameters(Ts &&... params) {
        prox<value_t>::parameters(std::forward<Ts>(params)...);
    }
    template <class Loss, class Logger, class Terminator>
    void solve(Loss &&loss, Logger &&logger, Terminator &&terminator) noexcept {
        if (params.verbose) 
            printf("%6s %10s\n", "Iter", "Fx");
        while (!terminator(k, fx, xb_c, xe_c, gb_c)) {
            step(std::forward<Loss>(loss));
            if(k % params.log_every == 0)
                logger(k, fx, xb_c, xe_c);
            if ((k % params.print_every == 0) && (params.verbose))
                printf("%6d %10.5f\n", k, fx);
            k++;
        }
    }
    template <class Loss, class Logger>
    void solve(Loss &&loss, Logger &&logger) noexcept {
        solve(std::forward<Loss>(loss), std::forward<Logger>(logger),
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
    int k, m_{1};
    value_t fx;
    vector_t x, y, g, gx, res, gxvec, resvec, xprox;
    value_t *xb, *xe, *gb;
    const value_t *xb_c, *xe_c, *gb_c;
    param_t params;
    
    template <class Loss>
    void step(Loss &&loss) noexcept {
        if (k == 0) {
            fx = loss(x.data(), g.data());
            auto stz = this->get_stepsize(std::forward<Loss>(loss), this,
                                          k, fx, xb_c, xe_c, gb_c);
            std::transform(x.begin(), x.end(), g.begin(), gx.begin(),
                           [stz](const value_t &xval, const value_t &gval) { return xval - stz * gval; }); 
            algebra::sub(gx, y, res);
            gxvec.insert(gxvec.begin(), gx.begin(), gx.end());
            resvec.insert(resvec.begin(), res.begin(), res.end());
            y = gx; // y1
            this->prox(y, x);
            fx = loss(x.data(), g.data());
        } else {
            auto mk = std::min(m_, k);
            auto stz = this->get_stepsize(std::forward<Loss>(loss), this,
                                          k, fx, xb_c, xe_c, gb_c);
            std::transform(x.begin(), x.end(), g.begin(), gx.begin(),
                           [stz](const value_t &xval, const value_t &gval) { return xval - stz * gval; });
            this->prox(gx, xprox);
            algebra::sub(gx, y, res);
            if (k < m_ + 1) {
                gxvec.insert(gxvec.end(), gx.begin(), gx.end());
                resvec.insert(resvec.end(), res.begin(), res.end());
            } else {
                auto idx = (k - 1) % m_;
                auto gxvec_b = gxvec.begin() + idx * x.size();
                auto resvec_b = resvec.begin() + idx * x.size();
                std::transform(gxvec_b, gxvec_b + x.size(), gx.begin(), gxvec_b,
                               [](const value_t &gxval, const value_t &gval) { return gval; }); 
                std::transform(resvec_b, resvec_b + x.size(), res.begin(), resvec_b,
                               [](const value_t &rxval, const value_t &rval) { return rval; });
            }
            assert(gxvec.size() == (mk + 1) * x.size());
            
            algebra::sub(xprox, x, res); // overwrite res
            auto t1 = fx + algebra::vdot(g, res);
            auto t2 = algebra::vdot(res, res) / (double) (2 * stz);

            aa_linear_solve(resvec, gxvec, mk + 1, params.reg, y.data());
            this->prox(y, x);
            fx = loss(x.data(), g.data());
            if (fx > t1 + t2) {
                y = gx;
                x = xprox;
                fx = loss(x.data(), g.data());
            } 
        }
    }

    void aa_linear_solve(const vector_t &resvec,
                         const vector_t &gxvec,
                         const int mem, 
                         const value_t reg,
                         value_t *out) {
        auto n = gxvec.size() / mem;     
        std::vector<value_t> rtr(mem * mem);
        blas_t::gemm('t', 'n', mem, mem, n, 1, resvec.data(),
                     n, resvec.data(), n, 0, rtr.data(), mem);
        auto normRR = algebra::ltwo(rtr); // upperbound for ||.||_2
        std::transform(rtr.begin(), rtr.end(), rtr.begin(),
                       [normRR](const value_t &val) { return val / normRR; });
        for (int i = 0; i < mem; i++) {
            rtr[i * mem + i] += reg;
        }
        std::vector<value_t> coeff(mem, 1.0);
        std::vector<int> ipiv(mem);
        auto failed = lapack_t::gesv(mem, 1, rtr.data(), mem, ipiv.data(), coeff.data(), mem);
        if (!failed) {
            auto sum = std::accumulate(coeff.begin(), coeff.end(), 0.0);
            std::transform(coeff.begin(), coeff.end(), coeff.begin(),
                           [sum](const value_t &val) { return val / sum; });
            blas_t::gemv('n', n, mem, 1, gxvec.data(), n, coeff.data(), 1, 0, out, 1);
        } else {
            std::cout << "Linear solver failed.\n";
            abort();
        }
    }
};

template <class value_t = double, template <class> class stepsize = stepsize::constant>
using anderson = optimizer::aaprox<value_t, stepsize, prox::none>;
} // namespace optimizer
#endif