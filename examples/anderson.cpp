#include <iostream>
#include "pomodoro.hpp"

using namespace function::loss;
using namespace utility;

int main(int argc, char *argv[]) {
    optimizer::aa_params<double> params;
    getargv<double>(argc, argv, params);

    std::string dataset = "australian";
    auto data = reader<double>::svm({"examples/data/" + dataset}, 690, 14);

    std::vector<double> x0(data.nfeatures());
    logistic_<double> loss(std::move(data));
    logger::value<double> logger;

    anderson<double> alg(params);
    double init_step = 0.0013751774503882516;
    alg.stepsize_parameters(init_step);
    const std::string filename = "_anderson_memory" + std::to_string(params.memory) 
                                 + "_step" + std::to_string(init_step);
    alg.initialize(x0);
    alg.solve(loss, logger);
    
    logger.csv("examples/output/" + dataset + filename + ".csv");
    std::cout << "optval: " << alg.getf() << '\n';
    return 0;
}
