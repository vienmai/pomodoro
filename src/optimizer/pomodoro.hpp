#ifndef OPTIMIZER_POMODORO_HPP_
#define OPTIMIZER_POMODORO_HPP_

#include "utility/null.hpp"
#include "utility/mpitype.hpp"

namespace optimizer {
template <class value_t, 
        template <class> class accelerator = accelerator::none,
        template <class> class prox = prox::none> 
struct pomodoro : public prox<value_t>,
                  public accelerator<value_t> {
    using prox_t = prox<value_t>;
    pomodoro() = default;
    pomodoro(admm_params<value_t> params) : params_(std::move(params)) {}

    void initialize(const std::vector<value_t> &x0) noexcept {
        k       = 0;
        rho     = params_.rho;
        elic    = params_.elic; 
        x       = x0;
        xloc    = x0;
        yloc    = std::vector<value_t>(x.size());
        rloc    = std::vector<value_t>(x.size());
        xb_c    = x.data();
        xe_c    = xb_c + x.size();
        yloc_c  = yloc.data();
        accelerator<value_t>::initialize(x0);
    };
    template <class... Ts> void accelerator_parameters(Ts &&... params) {
        accelerator<value_t>::parameters(std::forward<Ts>(params)...);
    }
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
    value_t fsend[2], frecv[2], rho, elic;
    std::vector<value_t> x, xloc, yloc, rloc;
    const value_t *xb_c, *xe_c, *yloc_c;
    admm_params<value_t> params_;

    template <class Subsolver, class Loss>
    void step(Subsolver &&subsolver, Loss &&loss, const MPI_Comm &COMM, const int rank) noexcept {
        subsolver.initialize(xloc); 
        loss.setinput(x, yloc, rho, -1); 
        fsend[0] = loss.getf(x);  // f(uloc)
        subsolver.solve(std::forward<Loss>(loss));
        xloc = subsolver.getx();
        MPI_Allreduce(xloc.data(), x.data(), xloc.size(), utility::MPI_Type<value_t>(), MPI_SUM, COMM);
        int nprocs = params_.nprocs; 
        std::transform(x.begin(), x.end(), x.begin(), 
                       [nprocs](const value_t &val) {return val / (double) nprocs; });
        algebra::sub(x, xloc, rloc);
        fsend[1] = algebra::vdot(rloc, rloc); 
        MPI_Allreduce(&fsend[0], &frecv[0], 2, utility::MPI_Type<value_t>(), MPI_SUM, COMM);

        auto yb      = yloc.begin();
        auto ye      = yloc.end();
        auto xloc_b  = xloc.begin();
        auto xb      = x.begin();
        while (yb != ye)
            *yb++ += - (rho - elic) * (*xloc_b++ - *xb++); // yloc <- vloc - ...
            
        this->accelerate(COMM, rank, k, xb_c, xe_c, yloc_c, x.data(), yloc.data()); // x <- uloc; yloc <- vloc
    }
};
} // namespace optimizer
#endif
