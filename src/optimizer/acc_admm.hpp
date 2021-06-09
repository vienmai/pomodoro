#ifndef OPTIMIZER_ACC_ADMM_HPP_
#define OPTIMIZER_ACC_ADMM_HPP_

#include "utility/null.hpp"
#include "utility/mpitype.hpp"

namespace optimizer {
template <class value_t, template <class> class prox = prox::none> 
struct proxadmm_ : public prox<value_t>{
    using prox_t = prox<value_t>;
    using blas_t = algebra::matrix::blas<value_t>;
    proxadmm_() = default;
    proxadmm_(admm_params<value_t> params) : params_(std::move(params)) {}

    void initialize(const std::vector<value_t> &x0) noexcept {
        k      = 0;
        rho    = params_.rho;  
        auto n = x0.size();                    
        x      = x0;
        xcur   = x0;
        uloc   = x0;
        u2p    = x0;
        xloc   = std::vector<value_t>(n);
        yloc   = std::vector<value_t>(n);
        vloc   = std::vector<value_t>(n);
        v2p    = std::vector<value_t>(n);
        ycur   = std::vector<value_t>(n);
        xb_c   = x.data();
        xe_c   = xb_c + n;
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
    int k;
    value_t fsend[2], frecv[2], rho;
    admm_params<value_t> params_;
    std::vector<value_t> x, xloc, yloc, uloc, vloc;
    std::vector<value_t> xcur, ycur, u2p, v2p;
    const value_t *xb_c, *xe_c;

    template <class Subsolver, class Loss>
    void step(Subsolver &&subsolver, Loss &&loss, const MPI_Comm &COMM, const int rank) noexcept {
        subsolver.initialize(xloc); 
        loss.setinput(uloc, vloc, rho, -1); 
        fsend[0] = loss.getf(uloc);
        // fsend[0] = loss.getf(x);
        subsolver.solve(std::forward<Loss>(loss));
        xloc = subsolver.getx();
        MPI_Allreduce(xloc.data(), x.data(), xloc.size(), utility::MPI_Type<value_t>(), MPI_SUM, COMM);
        int nprocs = params_.nprocs;
        std::transform(x.begin(), x.end(), x.begin(), [nprocs](const value_t &val) 
                       { return val / (value_t) nprocs; });
        auto yloc_b = yloc.begin();
        auto yloc_e = yloc.end();
        auto xloc_b = xloc.begin();
        auto vloc_b = vloc.begin();
        auto xb     = x.begin();
        while (yloc_b != yloc_e)
            *yloc_b++ = *vloc_b++ - rho * (*xloc_b++ - *xb++);
        auto alpha = k / (value_t) (k + 2);
        std::transform(xcur.begin(), xcur.end(), u2p.begin(), xcur.begin(),
                       [alpha](const value_t &xval, const value_t &uval) 
                       { return alpha * (-2 * xval + uval); });
        std::transform(ycur.begin(), ycur.end(), v2p.begin(), ycur.begin(),
                       [alpha](const value_t &yval, const value_t &vval) 
                       { return alpha * (-2 * yval + vval); });
        // std::transform(xcur.begin(), xcur.end(), u2p.begin(), xcur.begin(),
        //                [alpha](const value_t &xval, const value_t &uval) 
        //                { return  -alpha * xval; });
        // std::transform(ycur.begin(), ycur.end(), v2p.begin(), ycur.begin(),
        //                [alpha](const value_t &yval, const value_t &vval) 
        //                { return -alpha * yval; });
        u2p  = uloc; // store u_{k-1} before updating uloc
        v2p  = vloc;
        uloc = xcur;
        vloc = ycur;
        blas_t::axpy(x.size(), 1.0 + alpha, x.data(), 1, uloc.data(), 1);
        blas_t::axpy(yloc.size(), 1.0 + alpha, yloc.data(), 1, vloc.data(), 1);

        algebra::sub(xloc, x, xcur); // xcur <- xloc - x
        fsend[1] = algebra::vdot(xcur, xcur); // residual
        MPI_Allreduce(&fsend[0], &frecv[0], 2, utility::MPI_Type<value_t>(), MPI_SUM, COMM);
        xcur = x;
        ycur = yloc;
    }
};
template <class value_t = double>
using acc_admm = optimizer::proxadmm_<value_t, prox::none>;
} // namespace optimizer
#endif
