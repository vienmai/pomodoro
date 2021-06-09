#include <iostream>
#include <algorithm>
#include "algebra.hpp"
#include "matrix.hpp"
#include "function.hpp"
#include "prox.hpp"
#include "terminator.hpp"
#include "optimizer.hpp"
#include "utility.hpp"
#include <chrono>

using namespace function::loss;

int main(int argc, char *argv[]) {
    std::string dataset = "gisette";
    auto data =
        utility::reader<double>::svm({"test/data/" + dataset}, 6000, 5000);

    logistic_<double> loss(data);
    std::vector<double> x0(data.nfeatures());

    optimizer::proxgradient_params<double> params;
    utility::getargv<double>(argc, argv, params);
    prox::none<double> prox;
    optimizer::proxgradient_<double> alg(loss, params, prox);

    std::chrono::time_point<std::chrono::high_resolution_clock> tstart{
        std::chrono::high_resolution_clock::now()}, tend;
    
    // utility::logger::value<double> logger;
    // terminator::value<double, int> terminator(1E-3, 1E-8);
    alg.initialize(x0);
    alg.solve();

    tend = std::chrono::high_resolution_clock::now();
    auto telapsed =
        std::chrono::duration<double, std::chrono::seconds::period>(
            tend - tstart);
    std::cout << telapsed.count() << std::endl;

    // std::ofstream file("test/output/" + dataset + "_proxgrad" + ".csv");
    // if (file){   
    //     file << "k,t,f\n";
    //     for (const auto &log : logger)
    //         file << std::fixed << log.getk() << ',' << log.gett() << ',' << log.getf()
    //              << '\n';
    // }

    std::cout << "----------------------------------" << std::endl;
    std::cout << "optval: " << alg.getf() << '\n';
    // std::cout << "optsol: [ " ;
    // for (auto const &val : alg.getx())
    //     std::cout << val << ", ";
    // std::cout << "]" << std::endl;
    // std::cout << "----------------------------------" << std::endl;

    return 0;
}
