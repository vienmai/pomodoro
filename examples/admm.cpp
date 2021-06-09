#include <iostream>
#include <cstdlib> 
#include <algorithm>
#include <cassert>
#include "mpi.h"
#include "pomodoro.hpp"

using namespace function::loss;
using namespace utility;
template <class value_t>
data<value_t> getdata(const std::pair<std::string, std::array<size_t, 2> > &dataset,
                      int &m, const int rank, const int nprocs, const MPI_Comm &COMM);
static std::map<int, std::pair<std::string, std::array<size_t, 2> > > datasets{
    {1, {"heart", {270, 13}}},
    {2, {"australian", {690, 14}}},
    {3, {"a8a", {22696, 123}}},
    {4, {"mushrooms", {8124, 112}}},
    {5, {"dna", {2000, 180}}},
    {6, {"gisette", {6000, 5000}}},
    {7, {"w8a", {49749, 300}}},
    {8, {"real-sim", {72309, 20958}}},
    {9, {"diabetes", {768, 8}}}
};

int main(int argc, char *argv[]){
    int nprocs, rank;
    MPI_Comm COMM;
    MPI_Init(NULL, NULL);
    COMM = MPI_COMM_WORLD;
    MPI_Comm_size(COMM, &nprocs);
    MPI_Comm_rank(COMM, &rank);

    int m, dataset_choice{1};
    optimizer::admm_params<double> params;
    optimizer::lbfgs_params<double> subsolver_params;
    getargv(argc, argv, dataset_choice, params, subsolver_params);
    params.nprocs = nprocs;

    auto dataset = datasets.at(dataset_choice);
    auto dataloc = getdata<double>(dataset, m, rank, nprocs, COMM);
    admm_logistic<double> lossloc(dataloc);

#ifdef ADMM
    optimizer::pomodoro<double, accelerator::none> alg(params);
    const std::string solver = "_admm";
#elif defined APPA
    optimizer::pomodoro<double, accelerator::appa> alg(params);
    const std::string solver = "_appa";
#elif defined ANDERSON_ADMM
    optimizer::aa_admm<double> alg(params);
    const std::string solver = "_anderson_admm";
#elif defined ANDERSON
    optimizer::pomodoro<double, accelerator::anderson> alg(params);
    alg.accelerator_parameters(5, 1E-10);
    const std::string solver = "_aapomo";
#endif

    optimizer::lbfgs<double, stepsize::linesearch> subsolver(subsolver_params);
    logger::valfeas<double> logger;
    terminator::combine<double> terminator(params.max_iters, 1E-10, 1E-10, 1E-10, 1E-10);
    int n = dataloc.nfeatures();
    std::vector<double> x(n, 1 / (double) n);
    MPI_Bcast(&x[0], x.size(), MPI_DOUBLE, 0, COMM);

    alg.initialize(x);
    alg.solve(subsolver, lossloc, logger, terminator, COMM, rank);
    
    if (rank == 0) {
        [[maybe_unused]] auto &[dataname, dims] = dataset;
        logger.csv("examples/output/" + dataname + solver 
                   + "_nprocs" + std::to_string(nprocs) 
                   + "_rho" + std::to_string(params.rho) + ".csv");
    }
    
    MPI_Finalize();
    return 0;
}

template <class value_t>
data<value_t> getdata(const std::pair<std::string, std::array<size_t, 2> > &dataset,
                      int &m, const int rank,
                      const int nprocs, const MPI_Comm &COMM) {
    using namespace utility;
    std::shared_ptr<const matrix::amatrix<value_t>> A;
    auto b = std::make_shared<const std::vector<value_t>>();
    const value_t *Aptr;
    int n;
    if (rank == 0){
        const auto &[filename, dims] = dataset;
        auto data = reader<value_t>::svm({"examples/data/" + filename}, dims[0], dims[1]);
        m = data.nsamples();
        n = data.nfeatures();
        A = data.matrix();
        b = data.labels();
        Aptr = A->data();
    }
    MPI_Bcast(&m, 1, MPI_INT, 0, COMM);
    MPI_Bcast(&n, 1, MPI_INT, 0, COMM);

    int mloc;
    std::vector<value_t> aloc, bloc;
    int remainder = m % nprocs;
    if (remainder == 0) {
        mloc = m / nprocs;
        assert(mloc * nprocs == m);
        aloc = std::vector<value_t>(mloc * n);
        bloc = std::vector<value_t>(mloc);

        MPI_Scatter(b->data(), mloc, MPI_Type<value_t>(),
                    &bloc[0], mloc, MPI_Type<value_t>(), 0, COMM);

        MPI_Datatype sendtype;
        MPI_Type_vector(n, mloc, m, MPI_Type<value_t>(), &sendtype);
        MPI_Type_create_resized(sendtype, 0, mloc * sizeof(value_t), &sendtype);
        MPI_Type_commit(&sendtype);

        MPI_Scatter(Aptr, 1, sendtype, &aloc[0], mloc * n, MPI_Type<value_t>(), 0, COMM);

        MPI_Type_free(&sendtype);
    } else {
        std::vector<int> countloc(nprocs), offset(nprocs);
        int sum = 0, temp = m / nprocs;
        for (auto i = 0; i < nprocs; i++) {
            countloc[i] = (i < remainder) ? temp + 1 : temp;
            offset[i] = sum;
            sum += countloc[i];
        }
        assert(sum == m);
        mloc = countloc[rank];
        aloc = std::vector<value_t>(mloc * n);
        bloc = std::vector<value_t>(mloc);

        MPI_Scatterv(b->data(), countloc.data(), offset.data(), MPI_Type<value_t>(),
                    &bloc[0], mloc, MPI_Type<value_t>(), 0, COMM);

        MPI_Datatype sendtype, recvtype;
        MPI_Type_vector(n, 1, m, MPI_Type<value_t>(), &sendtype);
        MPI_Type_create_resized(sendtype, 0, sizeof(value_t), &sendtype);
        MPI_Type_commit(&sendtype);
        MPI_Type_vector(n, 1, mloc, MPI_Type<value_t>(), &recvtype);
        MPI_Type_create_resized(recvtype, 0, sizeof(value_t), &recvtype);
        MPI_Type_commit(&recvtype);

        MPI_Scatterv(Aptr, countloc.data(), offset.data(), sendtype, &aloc[0], mloc, recvtype, 0, COMM);

        MPI_Type_free(&sendtype);
        MPI_Type_free(&recvtype);
    }
    auto Aloc = std::make_shared<matrix::dmatrix<value_t>>(mloc, n, std::move(aloc));
    auto bloc_ = std::make_shared<const std::vector<value_t>>(std::move(bloc));
    return data<value_t>(Aloc, bloc_);
}
