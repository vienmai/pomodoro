#include <iostream>
#include <cstdlib>
#include "algebra.hpp"
#include "matrix.hpp"
#include "function.hpp"

using namespace function::loss;
using namespace matrix;

int main(int argc, char *argv[]) {

    // dmatrix<double> A(4, 4, {1, 2, 3, 4, 2, 1, 5, 6, 3, 5, 0, 8, 4, 6, 8, 9});
    // dmatrix<double> A(2, 4, {1, 2, 5, 6, 9, 10, 13, 14});
    // dmatrix<double> A(2, 4, { 3, 4, 7, 8, 11, 12, 15, 16});
    // dmatrix<double> A(4, 4, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    // dmatrix<double> A(2, 4, {0.23745072, 1.09557628, 1.38311306, -0.71729365, 
    //                         -0.10427326, -0.58578711, 0.50070193, -1.79345269});
    dmatrix<double> A(2, 4, {0.99426016, -1.19494335, 0.93119308, -0.26876015, 
                             0.92536737, -0.30543479, 0.80630967, -0.21159942});
    // dmatrix<double> A(4, 4, {0.23745072, 1.09557628, 0.99426016, -1.19494335, 
    //                          1.38311306, -0.71729365, 0.93119308, -0.26876015,
    //                          -0.10427326, -0.58578711, 0.92536737, -0.30543479, 
    //                          0.50070193, -1.79345269, 0.80630967, -0.21159942});

    srand(1234);

    // for (int i = 0; i < m; i++)
    // {
    //     for (int j = 0; j < n; j++)
    //     {
    //         A(i, j) = (rand() / (double)(RAND_MAX)) * 2 - 1;
    //     }
    // }

    std::cout << A << std::endl;

    std::ofstream outfile("A2x4_np_1.txt", std::ofstream::binary);
    A.save(outfile);
    
    // std::ifstream infile("A.txt", std::ifstream::binary);
    // A.load(infile);
    // std::vector<double> b{1, -2, 1};

    // data<double> data_(A, b);

    // int m = data_.nsamples();
    // int n = data_.nfeatures();
    // size_t m_ = 222099496726;

    // printf("m = %d,  n = %d\n", m, n);
    // printf("size_t(m) = %zu\n", size_t(m_));

    // std::vector<int> r{-1, -2, -3, 4, 5};
    // for (auto i : r)
    //     std::cout << i << " ";
    // std::cout << std::endl;

    // auto *ibegin = &r[0];
    // auto *iend = &r[3];

    // std::cout << "r[0] = " << *(ibegin) << std::endl;

    // int k=0;
    // auto *itemp{ibegin};
    // int *v;
    // v = (int *)malloc(sizeof(int)*3);
    // while (itemp != iend)
    // {   
    //     std::cout << k << std::endl;
    //     v[k++] = *(itemp++);
    // }
    
    
    // for (int j = 0; j < 4; j++)
    //     std::cout << v[j] <<" ";
    // std::cout << std::endl;

    // for (int i = 0; i < n; i++)
    // {
    //     for (int j = 0; j < m; j++)
    //         std::cout << A_[ i*m + j] << " ";
    //     std::cout << std::endl;
    // }

    return 0;
}
