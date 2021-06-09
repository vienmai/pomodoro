#include <iostream>
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
double randn_double() { return (rand() / (double)(RAND_MAX)) * 2 - 1; }

int main(int argc, char *argv[]){
    int nprocs, rank;
    MPI_Comm COMM;
    MPI_Datatype mytype, tmp;
    MPI_Init(NULL, NULL);
    COMM = MPI_COMM_WORLD;
    MPI_Comm_size(COMM, &nprocs);
    MPI_Comm_rank(COMM, &rank);

    /* Declare A here to avoid out of scope before calling MPI_Scatter.
       Also make_shared b so that all processes know what is b->data() */
    int m, n;
    std::shared_ptr<const matrix::amatrix<double>> A;
    std::shared_ptr<const std::vector<double>> b = std::make_shared<const std::vector<double>>();
    std::vector<double> x;
    const double *add;
    if (rank == 0){
        auto data = reader<double>::svm({"test/data/heart"}, 270, 13);
        m = data.nsamples();
        n = data.nfeatures();
        A = data.matrix();
        b = data.labels();
        A->memptr(&add);
        x = std::vector<double>(n);
    }
    MPI_Bcast(&m, 1, MPI_INT, 0, COMM);
    MPI_Bcast(&n, 1, MPI_INT, 0, COMM);
    
    const int mloc = m / nprocs;
    assert(mloc * nprocs == m);
    std::vector<double> aloc(mloc * n);
    std::vector<double> bloc(mloc);
    std::vector<double> xloc(n);
    std::vector<double> muloc(n);
    if (rank != 0){
        x = std::vector<double>(n);
    }
    MPI_Scatter(b->data(), mloc, MPI_DOUBLE, &bloc[0], mloc, MPI_DOUBLE, 0, COMM);
    MPI_Bcast(&x[0], x.size(), MPI_DOUBLE, 0, COMM);

    MPI_Type_vector(n, mloc, m, MPI_DOUBLE, &tmp);
    MPI_Type_commit(&tmp);
    MPI_Type_create_resized(tmp, 0, mloc*sizeof(double), &mytype);
    MPI_Type_commit(&mytype);

    MPI_Scatter(add, 1, mytype, &aloc[0], mloc*n, MPI_DOUBLE, 0, COMM);

    MPI_Type_free(&tmp);
    MPI_Type_free(&mytype);
    
    //at this point we should delete A? 
    matrix::dmatrix<double> Aloc(mloc, n, aloc);
    optimizer::admm_params<double> params;
    data<double> dataloc(Aloc, bloc);
    logistic_admm<double> lossloc(dataloc, params.rho);
    prox::none<double> prox;

    optimizer::proxgradient<double> alg(lossloc, params.subsolver_params, prox);
    
    xloc = std::vector<double>(std::begin(x), std::end(x));
    std::fill(std::begin(muloc), std::end(muloc), 0);
    double gamma = params.gamma;
    int max_iters = params.max_iters;
    for (int iter=0; iter < max_iters; iter++){
        alg.initialize(x);
        alg.solve();
        xloc = alg.getx();
        MPI_Allreduce(xloc.data(), x.data(), xloc.size(), MPI_DOUBLE, MPI_SUM, COMM);
        std::transform(std::begin(x), std::end(x), std::begin(x),
                       std::bind(std::multiplies<double>(), std::placeholders::_1, 1.0 / (double)nprocs));
        auto muloc_b = std::begin(muloc);
        auto muloc_e = std::end(muloc);
        auto xloc_b = std::begin(xloc);
        auto x_b = std::begin(x);
        while (muloc_b != muloc_e)
            *muloc_b++ += gamma * (*xloc_b++ - *x_b++);
    }
    if (rank==0){
        printvec(x);
    }
    MPI_Finalize();
    return 0;
}
