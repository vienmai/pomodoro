#include <iostream>
#include <ctime>   
#include <cstdlib> 
#include <algorithm>
#include "algebra.hpp"
#include "matrix.hpp"
#include "function.hpp"
#include "prox.hpp"
#include "optimizer.hpp"
#include "utility.hpp"
#include "mpi.h"

using namespace function::loss;
using namespace utility;
template<class value_t>
double randn_double() { return (rand() / (double)(RAND_MAX)) * 2 - 1; }

namespace optimizer{
template <class value_t> struct parameters{
    bool is_sparse;
    value_t l2_reg;
    int max_iters;
    value_t stepsize;
    bool linesearch;
    bool ls_adapt;
    value_t ls_ratio;
    value_t max_step;
    bool verbose;
    int print_every;
    value_t rho;
    value_t gamma;
    // anoptimizer<value_t> &subsolver;
    parameters() : is_sparse(0),
                   l2_reg(0.0),
                   max_iters(500),
                   stepsize(0.1),
                   linesearch(1),
                   ls_adapt(1),
                   ls_ratio(2.0),
                   max_step(1.0),
                   verbose(0),
                   print_every(1e6),
                   rho(0.1),
                   gamma(rho) {}
    ~parameters() = default;
};
}

int main(int argc, char *argv[]){
    int nprocs, rank;
    MPI_Comm COMM;
    MPI_Datatype mytype, tmp;
    MPI_Init(NULL, NULL);
    COMM = MPI_COMM_WORLD;
    MPI_Comm_size(COMM, &nprocs);
    MPI_Comm_rank(COMM, &rank);

    int m, n;
    const double *add;
    std::vector<double> x, b;
    // if (rank == 0)
    // {
    //     auto dataloc =
    //         utility::reader<double>::svm({"../test/data/heart"}, 270, 13);
    //     // A.memptr(&add);
    //     x = std::vector<double>(n);
    // }
    auto dataloc = reader<double>::svm({"../test/data/heart"}, 270, 13);
    // A.memptr(&add);
    m = dataloc.nsamples();
    n = dataloc.nfeatures();
    x = std::vector<double>(n);

    // MPI_Bcast(&m, 1, MPI_INT, 0, COMM);
    // MPI_Bcast(&n, 1, MPI_INT, 0, COMM);
    const int mloc = m / nprocs;
    assert(mloc * nprocs == m);

    std::vector<double> aloc(mloc * n);
    std::vector<double> bloc(mloc);
    std::vector<double> xloc(n);
    std::vector<double> muloc(n);
    if (rank != 0){
        x = std::vector<double>(n);
    }

    // MPI_Scatter(&b[0], mloc, MPI_DOUBLE,
    //             &bloc[0], mloc, MPI_DOUBLE, 0, COMM);
    // MPI_Bcast(&x[0], x.size(), MPI_DOUBLE, 0, COMM);

    // matrix::dmatrix<double> Aloc(mloc, n);
    // if (rank==0){
    //     std::ifstream file("A2x4_np_0.txt", std::ifstream::binary);
    //     Aloc.load(file);
    //     std::cout << Aloc << std::endl;
    // }else if (rank==1){
    //     std::ifstream file("A2x4_np_1.txt", std::ifstream::binary);
    //     Aloc.load(file);
    //     std::cout << Aloc << std::endl;
    // }

    // std::cout << "bloc = ";
    // printvec(bloc);

    // MPI_Type_vector(n, mloc, m, MPI_DOUBLE, &tmp);
    // MPI_Type_commit(&tmp);
    // MPI_Type_create_resized(tmp, 0, sizeof(double), &mytype);
    // MPI_Type_commit(&mytype);

    // MPI_Scatter(add, mloc * n, MPI_DOUBLE,
    //             &aloc[0], mloc * n, MPI_DOUBLE, 0, COMM);
    // MPI_Scatter(add, 1, mytype, &aloc[0], mloc*n, MPI_DOUBLE, 0, COMM);
    // MPI_Scatter(add, 1, mytype, Aloc_add, 1, mytype, 0, COMM);

    // if (rank == 0){
    //     // MPI_Send(add, mloc*n, MPI_DOUBLE, 1, 0, COMM);
    //     MPI_Send(add, 1, mytype, 1, 0, COMM);
    // }else{
    //     MPI_Recv(&aloc[0], mloc*n, MPI_DOUBLE, 0, 0, COMM, MPI_STATUS_IGNORE);
    // }

    // MPI_Barrier (COMM);
    // // printf("Rank %d\n", rank);
    // printvec(aloc);
    
    // matrix::dmatrix<double> Aloc(mloc, n, aloc);
    // const double *Aloc_add;
    // Aloc.memptr(&Aloc_add);
    // std::cout << "Aloc Proc: " << rank << std::endl;
    // std::cout << Aloc << std::endl;
    
    
    //at this point we should delete A? share_ptr auto delete when out of scope

    // data<double> dataloc(Aloc, bloc);
    logistic_admm<double> lossloc(dataloc, 0.1);
    prox::none<double> prox;

    optimizer::parameters<double> params;
    optimizer::proxgradient<double> alg(lossloc, params, prox);
    
    xloc = std::vector<double>(std::begin(x), std::end(x));
    std::fill(std::begin(muloc), std::end(muloc), 0);
    double gamma = params.gamma;
    int max_iters = params.max_step;
    for (int iter=0; iter < 20*max_iters; iter++){
        alg.initialize(x);
        alg.solve();
        xloc = alg.getx();
        MPI_Allreduce(xloc.data(), x.data(), xloc.size(), MPI_DOUBLE, MPI_SUM, COMM);

        auto muloc_b = std::begin(muloc);
        auto muloc_e = std::end(muloc);
        auto xloc_b = std::begin(xloc);
        auto x_b = std::begin(x);
        while (muloc_b != muloc_e)
            *muloc_b++ += gamma * (*xloc_b++) - (gamma/(double)nprocs) * (*x_b++);
    }
    if (rank==0){
        printvec(x);
    }
    MPI_Finalize();
    return 0;
}

