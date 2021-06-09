#ifndef OPTIMIZER_ADMM_HPP_
#define OPTIMIZER_ADMM_HPP_

#include "utility/null.hpp"
#include "utility/mpitype.hpp"
#include "utility/printer.hpp"

namespace optimizer {
template <class value_t, template <class> class prox = prox::none> 
struct proxadmm : public prox<value_t>{
    using prox_t = prox<value_t>;
    proxadmm() = default;
    proxadmm(admm_params<value_t> params) : params_(std::move(params)) {}

    void initialize(const std::vector<value_t> &x0) noexcept {
        k       = 0;
        rho     = params_.rho;
        x       = x0;
        xloc    = x0;
        muloc   = std::vector<value_t>(x.size());
        rloc    = std::vector<value_t>(x.size());
        xb_c    = x.data();
        xe_c    = xb_c + x.size();
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
               const MPI_Comm &COMM, const int rank) noexcept{
        solve(std::forward<Subsolver>(subsolver), std::forward<Loss>(loss),
              utility::detail::null{}, std::forward<Terminator>(terminator), COMM, rank);
    }
    template <class Subsolver, class Loss>
    void solve(Subsolver &&subsolver, Loss &&loss, const MPI_Comm &COMM, const int rank) noexcept{
        solve(std::forward<Subsolver>(subsolver), std::forward<Loss>(loss),
              utility::detail::null{}, terminator::iteration<value_t>{params_.max_iters}, COMM, rank);
    }

    std::vector<value_t> getx() const noexcept { return x; }
    value_t getf() const noexcept { return frecv[0]; }

private:
    int k;
    value_t fsend[2], frecv[2], rho;
    admm_params<value_t> params_;
    std::vector<value_t> x, xloc, muloc, rloc;
    const value_t *xb_c, *xe_c;

    template <class Subsolver, class Loss>
    void step(Subsolver &&subsolver, Loss &&loss, const MPI_Comm &COMM, const int rank) noexcept {
        subsolver.initialize(xloc);
        loss.setinput(x, muloc, rho, 1);
        fsend[0] = loss.getf(x);
        subsolver.solve(std::forward<Loss>(loss));
        xloc = subsolver.getx();
        MPI_Allreduce(xloc.data(), x.data(), xloc.size(), utility::MPI_Type<value_t>(), MPI_SUM, COMM);
        int nprocs = params_.nprocs; 
        std::transform(std::begin(x), std::end(x), std::begin(x), 
                       [nprocs](const value_t &val) {return val / (double) nprocs; });
        auto muloc_b = std::begin(muloc);
        auto muloc_e = std::end(muloc);
        auto xloc_b = std::begin(xloc);
        auto x_b = std::begin(x);
        while (muloc_b != muloc_e)
            *muloc_b++ += rho * (*xloc_b++ - *x_b++);

        algebra::sub(x, xloc, rloc);
        fsend[1] = algebra::vdot(rloc, rloc); 
        MPI_Allreduce(&fsend[0], &frecv[0], 2, utility::MPI_Type<value_t>(), MPI_SUM, COMM);
    }
};
template <class value_t = double>
using admm = optimizer::proxadmm<value_t, prox::none>;
} // namespace optimizer
#endif
