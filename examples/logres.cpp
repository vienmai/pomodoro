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

int main(int argc, char *argv[]){
    int m = 5000, n = 5000;
    std::vector<double> xopt(n);
    xopt[0] = 1;
    xopt[1] = 0.5;
    xopt[2] = -0.5;
    
    auto A = matrix::randn<double>(m, n); 
    auto b = utility::randn<double>(m*n*2);

    const double *aptr = A.data();
    const double *aptr_, *bptr_;

    utility::timer t;
    auto A_ = A;
    aptr_ = A_.data();
    t.elapsed();
    auto B_ = std::move(A);
    bptr_ = B_.data();
    t.elapsed();

    aptr = A.data();
    std::cout << aptr << " " << aptr_ << " " << bptr_ << " " << aptr << '\n';

    std::vector<double> x = utility::randn<double>(m * n * 2);
    for (int i=0; i< 100; i++){
        std::vector<double> y(x.size());
        std::transform(x.begin(), x.end(), y.begin(), [](const auto &val){ return 2 * val; });
    }
    t.elapsed();

    // std::vector<double> z(x.size());
    // for (int i=0; i< 100; i++){
    //     std::transform(x.begin(), x.end(), z.begin(), [](const auto &val) { return 2 * val; });
    // }
    // t.elapsed();

    // double noise = 0.01;
    // A.mult_add('n', 1, &xopt[0], noise, &b[0]);
    // std::transform(b.begin(), b.end(), b.begin(),
    //                [](const double val) { return 1.0 / (1.0 + std::exp(-val)); });
    // std::transform(b.begin(), b.end(), b.begin(),
    //                [](const double val) { return std::round(val); });
    // data<double> data(std::move(A), std::move(b));

    // std::vector<double> x0(data.nfeatures());
    // logistic_<double> loss(std::move(data));

    // optimizer::proxgradient_params<double> params;
    // getargv<double>(argc, argv, params);
    // optimizer::gradient_descent<double> alg(params);
    // logger::value<double> logger;

    // alg.initialize(x0);
    // alg.solve(loss, logger);


    // std::cout << "----------------------------------\n";
    // std::cout << "optval: " << alg.getf() << '\n';
    return 0;
}
