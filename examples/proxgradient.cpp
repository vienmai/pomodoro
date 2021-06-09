#include <iostream>
#include "pomodoro.hpp"

using namespace function::loss;
using namespace utility;

int main(int argc, char *argv[]) {
    optimizer::params<double> optim_params;
    optimizer::stepsize_params<double> step_params;
    getargv<double>(argc, argv, optim_params, step_params);

    std::string dataset = "australian";
    auto data = reader<double>::svm({"examples/data/" + dataset}, 690, 14);
    // auto data = reader<double>::svm({"examples/data/" + dataset}, 22696, 123);

    std::vector<double> x0(data.nfeatures());
    logistic_<double> loss(std::move(data));
    logger::value<double> logger;

#ifdef CONSTANT
    gradient_descent<double, stepsize::constant> alg(optim_params);
    // double stz = 1.3866002978708885e-07; // heart
    double stz = 0.0013751774503882516; // australian
    alg.stepsize_parameters(stz);
    const std::string filename = "_gradient_descent_constant_" + std::to_string(stz);
#elif defined LINESEARCH
    gradient_descent<double, stepsize::linesearch> alg(optim_params);
    double init_step = 1.0;
    alg.stepsize_parameters(init_step, step_params.max_step,
                            step_params.ls_adapt, step_params.ls_ratio);
    const std::string filename = "_gradient_descent_linesearch_initstep" + std::to_string(init_step);
// #elif defined ANDERSON
//     optimizer::aa_params<double> aa_params;
//     anderson<double> alg(aa_params);
//     double init_step = 0.0013751774503882516;
//     alg.stepsize_parameters(init_step);
//     const std::string filename = "_anderson" + std::to_string(aa_params.memory) + '_' + std::to_string(init_step);
#elif defined NESTEROV_CONSTANT
    nesterov<double, stepsize::constant> alg(optim_params);
    double stz = 0.0013751774503882516;
    alg.stepsize_parameters(stz);
    const std::string filename = "_nesterov_constant_" + std::to_string(stz);
#elif defined NESTEROV_LINESEARCH
    nesterov<double, stepsize::linesearch> alg(optim_params);
    double init_step = 1.0;
    alg.stepsize_parameters(init_step, step_params.max_step,
                            step_params.ls_adapt, step_params.ls_ratio);
    const std::string filename = "_nesterov_linesearch_initstep" + std::to_string(init_step);
#endif
    
    alg.initialize(x0);
    alg.solve(loss, logger);

    logger.csv("examples/output/" + dataset + filename + ".csv");
    std::cout << "optval: " << alg.getf() << '\n';
    return 0;
}
