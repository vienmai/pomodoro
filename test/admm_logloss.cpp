#include <iostream>
#include "algebra.hpp"
#include "matrix.hpp"
#include "function.hpp"

using namespace function::loss;
using namespace matrix;

int main(int argc, char *argv[]) {

    dmatrix<double> A(3, 3, {1., 0., 3., 0., 1., 0., 2., 0., 1.});
    std::vector<double> b{1, -2, 1};
    std::cout << A << std::endl;

    data<double> data(A, b);
    const std::vector<double> x0{1, -2, -5};

    logistic<double> loss(data);
    double fval = loss(&x0[0]);
    std::cout << "f(0) = " << fval << std::endl; 

    std::vector<double> x{1, -1, -1.5};
    std::vector<double> mu{0.5, 0.5, 0.0};

    double *subgrad = new double[3];
    logistic_admm<double> subloss(data, 0.5);
    // double subfval = subloss(&x0[0], subgrad, mu, x);
    // std::cout << "f(0) = " << subfval << std::endl; // Expected

    double *grad = new double[3];
    // double fval_ = loss(&x0[0], grad);  // 
    // std::cout << "f(0) = " << fval_ << std::endl;
    loss.grad(&x0[0], grad);

    subloss.grad(&x0[0], subgrad, mu, x); // Expected (0.0, array([-7.58256136e-10,  8.49670851e-18, -1.51651212e-09]))
    for (int i = 0; i < 3; i++)
        std::cout << grad[i] <<  "   " << subgrad[i] << " ";
    std::cout << std::endl;

    delete[] grad; 
    delete[] subgrad;
    return 0;
}
