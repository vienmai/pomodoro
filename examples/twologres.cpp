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
    int m = 6000, n = 5000;

    auto A = matrix::randn<double>(m, n); 
    auto xopt = utility::randn<double>(n);
    auto b = utility::randn<double>(m);

    double noise = 0.01;
    A.mult_add('n', 1, &xopt[0], noise, &b[0]);
    std::transform(b.begin(), b.end(), b.begin(),
                   [](const double val) { return 1.0 / (1.0 + std::exp(-val)); });
    std::transform(b.begin(), b.end(), b.begin(),
                   [](const double val) { return std::round(val); });
    data<double> data(std::move(A), std::move(b));

    std::vector<double> x0 = xopt;
    logistic<double> loss(data);
    logistic_<double> loss_(data);

    std::vector<double> g(n); // gradient vector
    double fx, fx_;

    utility::timer t;
    std::vector<double> x = x0;
    for (int i=0; i < 200; i++){
        fx = loss(x.data(), g.data());
        std::transform(x.begin(), x.end(), x.begin(), [](const double val) { return 1.2 * val; });
    }
    t.elapsed();

    std::vector<double> x_ = x0;
    for (int i=0; i < 200; i++){
        fx_ = loss_(x_.data(), g.data());
        std::transform(x_.begin(), x_.end(), x_.begin(), [](const double val) { return 1.2 * val; });
    }
    t.elapsed();

    std::cout << "fx = " << fx << " fx_ = " << fx_ << '\n';

    return 0;
}
