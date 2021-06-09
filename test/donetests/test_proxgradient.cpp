#include <iostream>
#include <algorithm>
#include "algebra.hpp"
#include "matrix.hpp"
#include "function.hpp"
#include "prox.hpp"
#include "terminator.hpp"
#include "optimizer.hpp"
#include "utility.hpp"

using namespace function::loss;

int main(int argc, char *argv[]) {
    auto data =
        utility::reader<double>::svm({"test/data/heart"}, 270, 13);
        // utility::reader<double>::svm({"test/data/gisette_scale"}, 6000, 5000);
    logistic<double> loss(data);
    prox::none<double> prox;

    std::vector<double> x0(data.nfeatures());

    optimizer::proxgradient_params<double> params;
    utility::getargv<double>(argc, argv, params);
    std::cout << params.stepsize << std::endl;
    optimizer::proxgradient<double> alg(loss, params, prox);

    utility::logger::value<double, int> logger;
    // terminator::value<double, int> terminator(1E-3, 1E-8);
    alg.initialize(x0);
    alg.solve();
    // alg.solve(logger, terminator);
    
    std::ofstream file("terminator.csv");
    if (file){   
        file << "k,t,f\n";
        for (const auto &log : logger)
            file << std::fixed << log.getk() << ',' << log.gett() << ',' << log.getf()
                 << '\n';
    }

    std::cout << "----------------------------------" << std::endl;
    std::cout << "optval: " << alg.getf() << '\n';
    // std::cout << "optsol: [ " ;
    // for (auto const &val : alg.getx())
    //     std::cout << val << ", ";
    // std::cout << "]" << std::endl;
    // std::cout << "----------------------------------" << std::endl;

    return 0;
}
