#include <iostream>
#include "pomodoro.hpp"

using namespace function::loss;
using namespace utility;

int main(int argc, char *argv[]){
    std::string dataset = "australian";
    auto data = reader<double>::svm({"examples/data/" + dataset}, 690, 14);

    std::vector<double> x0(data.nfeatures());
    logistic_<double> loss(std::move(data));

    optimizer::lbfgs_params<double> params;
    getargv<double>(argc, argv, params);
    logger::value<double> logger;

    lbfgs<double, stepsize::linesearch> alg(params);
    alg.initialize(x0);
    alg.solve(loss, logger);

    const std::string filename = "_lbfgs_memory_" + std::to_string(params.memory);
    logger.csv("examples/output/" + dataset + filename + ".csv");
    
    std::cout << "optval: " << alg.getf() << '\n';
    return 0;
}
