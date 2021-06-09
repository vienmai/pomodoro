#ifndef OPTIMIZER_ANDERSON_ADMM_HPP_
#define OPTIMIZER_ANDERSON_ADMM_HPP_

#include "utility/null.hpp"
#include "utility/mpitype.hpp"

namespace optimizer {
template <class value_t, template <class> class prox = prox::none> 
struct anderson_admm : public prox<value_t> {
    using vector_t = std::vector<value_t>;
    using blas_t = algebra::matrix::blas<value_t>;
    using lapack_t = algebra::matrix::lapack<value_t>;
    anderson_admm() = default;
    anderson_admm(admm_params<value_t> params) : params_(std::move(params)) {}

    void initialize(const std::vector<value_t> &x0) noexcept {
        k      = 0;
        rho    = params_.rho;
        elic    = params_.elic;
        m_     = params_.memory;
        auto n = x0.size();                    
        x      = x0;
        uloc   = x0;
        xloc   = std::vector<value_t>(n);
        yloc   = std::vector<value_t>(n);
        vloc   = std::vector<value_t>(n);
        res    = vector_t(n);
        xb_c   = x.data();
        xe_c   = xb_c + n;
        gxvec.reserve(n * (m_ + 1));
        gyvec.reserve(n * (m_ + 1));
        resvec.reserve(n * (m_ + 1));
    };
    template <class Subsolver, class Loss, class Logger, class Terminator>
    void solve(Subsolver &&subsolver, Loss &&loss, Logger &&logger,
               Terminator &&terminator, const MPI_Comm &COMM, const int rank) noexcept{
        if ((rank == 0) && (params_.verbose))
            printf("%8s %10s %10s\n", "Iter", "Fx", "Feas");
        while (!std::forward<Terminator>(terminator)(k, frecv[0], sqrt(frecv[1]), xb_c, xe_c)){
            step(std::forward<Subsolver>(subsolver), std::forward<Loss>(loss), COMM, rank);
            if (k % params_.log_every == 0){
                std::forward<Logger>(logger)(k, frecv[0], sqrt(frecv[1]), xb_c, xe_c);
                if ((rank == 0) && (params_.verbose))
                    printf("%8d %10.5f %10.5f \n", k, frecv[0], frecv[1]);
            }
            k++;
        }
    }
    template <class Subsolver, class Loss, class Terminator>
    void solve(Subsolver &&subsolver, Loss &&loss, Terminator &&terminator,
               const MPI_Comm &COMM, const int rank) noexcept {
        solve(std::forward<Subsolver>(subsolver), std::forward<Loss>(loss),
              utility::detail::null{}, std::forward<Terminator>(terminator), COMM, rank);
    }
    template <class Subsolver, class Loss>
    void solve(Subsolver &&subsolver, Loss &&loss, const MPI_Comm &COMM, const int rank) noexcept {
        solve(std::forward<Subsolver>(subsolver), std::forward<Loss>(loss),
              utility::detail::null{}, terminator::iteration<value_t>{params_.max_iters}, COMM, rank);
    }
    std::vector<value_t> getx() const noexcept { return x; }
    value_t getf() const noexcept { return frecv[0]; }

private:
    int k, m_{1};
    value_t fsend[2], frecv[2], rho, elic;
    admm_params<value_t> params_;
    std::vector<value_t> x, xloc, yloc, uloc, vloc;
    std::vector<value_t> res, gxvec, gyvec, resvec;
    const value_t *xb_c, *xe_c;

    template <class Subsolver, class Loss>
    void step(Subsolver &&subsolver, Loss &&loss, const MPI_Comm &COMM, const int rank) noexcept {
        subsolver.initialize(xloc);
        loss.setinput(uloc, vloc, rho, -1);
        fsend[0] = loss.getf(uloc);
        subsolver.solve(std::forward<Loss>(loss));
        xloc = subsolver.getx();
        MPI_Allreduce(xloc.data(), x.data(), xloc.size(), utility::MPI_Type<value_t>(), MPI_SUM, COMM);
        int nprocs = params_.nprocs;
        std::transform(x.begin(), x.end(), x.begin(), [nprocs](const value_t &val) { return val / (value_t)nprocs; });
        algebra::sub(xloc, x, res);         // res <- xloc - x
        fsend[1] = algebra::vdot(res, res); // residual
        MPI_Allreduce(&fsend[0], &frecv[0], 2, utility::MPI_Type<value_t>(), MPI_SUM, COMM);

        auto yloc_b = yloc.begin();
        auto yloc_e = yloc.end();
        auto xloc_b = xloc.begin();
        auto vloc_b = vloc.begin();
        auto xb = x.begin();
        while (yloc_b != yloc_e)
            *yloc_b++ = *vloc_b++ - (rho - elic) * (*xloc_b++ - *xb++);

        algebra::add(x, yloc, res); // res <- x + y
        algebra::sub(res, uloc, res);
        algebra::sub(res, vloc, res); // r0 = res = x + y - u -v
        if (k == 0) {
            gxvec.insert(gxvec.begin(), x.begin(), x.end()); // g(x0)
            gyvec.insert(gyvec.begin(), yloc.begin(), yloc.end()); // g(x0)
            resvec.insert(resvec.begin(), res.begin(), res.end()); // r0
            uloc = x;    // x1 = g(x0)
            vloc = yloc; // x1 = g(x0)
        } else {
            auto mk = std::min(m_, k);
            if (k < m_ + 1) {
                gxvec.insert(gxvec.end(), x.begin(), x.end());
                gyvec.insert(gyvec.begin(), yloc.begin(), yloc.end()); // g(xk)
                resvec.insert(resvec.end(), res.begin(), res.end());
            } else {
                auto idx = m_ > 0 ? (k - 1) % m_ : 0;
                auto offset = idx * x.size();
                std::copy(x.begin(), x.end(), gxvec.begin() + offset);
                std::copy(yloc.begin(), yloc.end(), gyvec.begin() + offset);
                std::copy(res.begin(), res.end(), resvec.begin() + offset);
            }
            assert(gxvec.size()  == (mk + 1) * x.size());
            assert(gyvec.size()  == (mk + 1) * yloc.size());
            assert(resvec.size() == (mk + 1) * res.size());
            
            auto mem = mk + 1;
            auto n = x.size();
            std::vector<value_t> rtr_send(mem * mem);
            std::vector<value_t> rtr_recv(mem * mem);
            blas_t::gemm('t', 'n', mem, mem, n, 1, resvec.data(),
                         n, resvec.data(), n, 0, rtr_send.data(), mem);
            MPI_Allreduce(rtr_send.data(), rtr_recv.data(), rtr_send.size(),
                          utility::MPI_Type<value_t>(), MPI_SUM, COMM);
            auto normRR = algebra::ltwo(rtr_recv);
            std::transform(rtr_recv.begin(), rtr_recv.end(), rtr_recv.begin(),
                           [normRR](const value_t &val) { return val / normRR; });
            for (int i = 0; i < mem; i++) {
                rtr_recv[i * mem + i] += params_.aa_reg;
            }
            aa_linear_solve(rtr_recv, gxvec, gyvec, mem, params_.aa_reg, uloc.data(), vloc.data());
        }
    }
    void aa_linear_solve(vector_t &rtr, const vector_t &gxvec, const vector_t &gyvec,
                         const int mem, const value_t reg, value_t *out1, value_t *out2) {
        auto n = gxvec.size() / mem;     
        std::vector<value_t> coeff(mem, 1.0);
        std::vector<int> ipiv(mem);
        auto failed = lapack_t::gesv(mem, 1, rtr.data(), mem, ipiv.data(), coeff.data(), mem);
        if (!failed) {
            value_t sum = std::accumulate(coeff.begin(), coeff.end(), 0.0);
            std::transform(coeff.begin(), coeff.end(), coeff.begin(),
                           [sum](const value_t &val) { return val / sum; });
            blas_t::gemv('n', n, mem, 1, gxvec.data(), n, coeff.data(), 1, 0, out1, 1);
            blas_t::gemv('n', n, mem, 1, gyvec.data(), n, coeff.data(), 1, 0, out2, 1);
        } else {
            std::cout << "Linear solver failed.\n";
            abort();
        }
    }
};
template <class value_t = double>
using aa_admm = optimizer::anderson_admm<value_t, prox::none>;
} // namespace optimizer
#endif
