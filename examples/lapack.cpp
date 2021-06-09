#include <iostream>
#include <algorithm>
#include "algebra.hpp"
#include "matrix.hpp"
#include "function.hpp"
#include "prox.hpp"
#include "terminator.hpp"
#include "optimizer.hpp"
#include "utility.hpp"

using namespace matrix;

int main(int argc, char *argv[]){

    // auto x = utility::randn<double>(5);
    // utility::printvec(x);
    // auto y = utility::randn<double>(5);
    // utility::printvec(y);
    // algebra::sub(x.data(), y, x);
    // utility::printvec(x);;


    int m = 3, n = 4;
    // auto A = matrix::randn<double>(m, n);
    dmatrix<double> A(4, 3, {0.23745072, 1.09557628, 0.99426016, -1.19494335,
                             1.38311306, -0.71729365, 0.93119308, -0.26876015,
                             -0.10427326, -0.58578711, 0.92536737, -0.30543479});

    std::vector<double> coeff{-0.089, 1.254, -3.105};
    std::vector<double> out(n);
    A.mult_add('n', 1, coeff.data(), 0, out.data());

    utility::printvec(out);

    std::vector<double> cvec{1, 0, 0, 0, 1, 0, 0, 0, 1};
    dmatrix<double> C(3, 3, cvec);
    const double *Aptr = A.data();
    
    std::cout << A << '\n';

    // gemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
    algebra::matrix::blas<double>::gemm('t', 'n', m, m, n, 1, Aptr,
                                         n, Aptr, n, 0, cvec.data(), m); // A.T.dot(A)

    for (int i = 0; i < m; i++) {
        cvec[i * m + i] += 1.515;
    }

    std::cout << C << '\n';
    dmatrix<double> D(3, 3, cvec);
    std::cout << D << '\n';
    std::vector<double> b(m, 1);
    int ipiv[m];
    algebra::matrix::lapack<double>::gesv(m, 1, cvec.data(), m, ipiv, b.data(), m);
    
    utility::printvec(b);

    return 0;
}
