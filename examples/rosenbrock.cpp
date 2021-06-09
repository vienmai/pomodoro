#include <iostream>
#include "pomodoro.hpp"

using namespace function::loss;
using namespace utility;

int main(int argc, char *argv[]){
    std::vector<double> x0{-1.2, 1};
    optimizer::proxgradient_params<double> params;
    getargv<double>(argc, argv, params);
    logger::value<double> logger;

    gradient_descent<double, stepsize::linesearch> alg(params);
    alg.stepsize_parameters(1, 10, 1, 2);
    alg.initialize(x0);
    alg.solve(rosenbrock(), logger);
    
    const std::string filename = "rosenbrock_gradient_descent_linesearch";
    logger.csv("test/output/" + filename + ".csv");
    
    std::cout << "xsol = [ ";
    for (auto const &val : alg.getx())
        std::cout << val << " ";
    std::cout << "]\n";
    
    return 0;
}
