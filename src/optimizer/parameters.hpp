#ifndef OPTIMIZER_PARAMETERS_HPP_
#define OPTIMIZER_PARAMETERS_HPP_

namespace optimizer{
template <class value_t> struct params {
    int max_iters;
    value_t epsilon;
    bool verbose;
    int print_every;
    int log_every;
    params() : max_iters(100),
               epsilon(1e-10),
               verbose(0),
               print_every(1),
               log_every(print_every) {}
    virtual ~params() = default;
};    
template <class value_t> 
struct proxgradient_params : public params<value_t> {
    value_t stepsize;
    bool linesearch;
    bool ls_adapt;
    value_t ls_ratio;
    value_t max_step;
    proxgradient_params() : stepsize(1.0),
                            linesearch(1),
                            ls_adapt(1),
                            ls_ratio(2.0),
                            max_step(10.0){}
};
template <class value_t>
struct aa_params : public params<value_t>
{
    int memory;
    value_t reg;
    aa_params() : memory(5),
                  reg(1E-10) {}
};
template <class value_t>
struct lbfgs_params : public params<value_t> {
    int memory;
    value_t init_step;
    bool is_ls;
    bool ls_adapt;
    value_t ls_ratio;
    value_t max_step;
    value_t ls_rho;
    value_t ls_dec;
    int ls_maxiter;
    lbfgs_params() : memory(5),
                     init_step(1.0),
                     is_ls(true),
                     ls_adapt(false),
                     ls_ratio(2.0),
                     max_step(10),
                     ls_rho(0.5),
                     ls_dec(0.05),
                     ls_maxiter(20) {}
};
template <class value_t> 
struct admm_params : public params<value_t> {
    value_t rho;
    value_t elic;
    value_t aa_reg;
    int nprocs;
    int memory;
    admm_params() : rho(1),
                    elic(0),
                    aa_reg(1E-10),
                    nprocs(1),
                    memory(5) {}
};
template <class value_t> struct stepsize_params {
    value_t init_step;
    bool is_ls;
    bool ls_adapt;
    value_t ls_ratio;
    value_t max_step;
    value_t fmin;
    stepsize_params() : init_step(1.0),
                        is_ls(true),
                        ls_adapt(1),
                        ls_ratio(2.0),
                        max_step(10.0),
                        fmin(0) {}
};
}// namespace optimizer
#endif



   