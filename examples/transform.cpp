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
    int m = 4, n = 3;
    std::vector<double> x(n * m);
    
    auto b = utility::randn<double>(m);

    utility::printvec(b);

    auto idx = 2;
    auto x_start = x.begin() + idx * m;
    std::transform(x_start, x_start + m, b.begin(), x_start,
                   [](const double gxval, const double gval) { return gval; });
    utility::printvec(x);
    
    return 0;
}
