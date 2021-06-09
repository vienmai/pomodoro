#include <iostream>
#include <cstdlib>
#include "mpi.h"
#include "pomodoro.hpp"

using namespace function::loss;
using namespace matrix;
using namespace utility;

template <class value_t>
void getdata(int &m, int &n, int &mloc,
             std::vector<value_t> &aloc, std::vector<value_t> &bloc,
             const int rank, const int nprocs, const MPI_Comm COMM);

int main(int argc, char *argv[]) {
    int nprocs, rank;
    MPI_Comm COMM;
    MPI_Init(NULL, NULL);
    COMM = MPI_COMM_WORLD;
    MPI_Comm_size(COMM, &nprocs);
    MPI_Comm_rank(COMM, &rank);

    if (rank==0){
        dmatrix<double> A(4, 4, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        std::ofstream outfile("A4x4.txt", std::ofstream::binary);
        A.save(outfile);

        std::cout << A << std::endl;
    }

    int m, n, mloc;
    std::vector<double> aloc, bloc;
    getdata<double>(m, n, mloc, aloc, bloc, rank, nprocs, COMM);

    matrix::dmatrix<double> Aloc(mloc, n, std::move(aloc));
    std::cout << "Aloc " << rank <<  "\n" << Aloc << std::endl;
    
    MPI_Finalize();
    return 0;
}

template <class value_t>
void getdata(int &m, int &n, int &mloc,
             std::vector<value_t> &aloc, std::vector<value_t> &bloc,
             const int rank, const int nprocs, const MPI_Comm COMM){
    matrix::dmatrix<value_t> A;
    const value_t *Aptr;
    if (rank == 0){
        std::ifstream infile("A4x4.txt", std::ifstream::binary);
        A.load(infile);
        m = A.nrows();
        n = A.ncols();
        A.memptr(&Aptr);
    }
    MPI_Bcast(&m, 1, MPI_INT, 0, COMM);
    MPI_Bcast(&n, 1, MPI_INT, 0, COMM);

    int remainder = m % nprocs;
    int countloc[nprocs], offset[nprocs];
    int sum = 0, temp = m / nprocs;
    for (auto i = 0; i < nprocs; i++){
        countloc[i] = (i < remainder) ? temp + 1 : temp;
        offset[i] = sum;
        sum += countloc[i];
    }

    assert(sum == m);
    mloc = countloc[rank];
    aloc = std::vector<value_t>(mloc * n);

    MPI_Datatype sendtype, recvtype;
    MPI_Type_vector(n, 1, m, MPI_Type<value_t>(), &sendtype);
    MPI_Type_create_resized(sendtype, 0, sizeof(value_t), &sendtype);
    MPI_Type_commit(&sendtype);
    MPI_Type_vector(n, 1, mloc, MPI_Type<value_t>(), &recvtype);
    MPI_Type_create_resized(recvtype, 0, sizeof(value_t), &recvtype);
    MPI_Type_commit(&recvtype);

    MPI_Scatterv(Aptr, countloc, offset, sendtype, &aloc[0], mloc, recvtype, 0, COMM);
    
    MPI_Type_free(&sendtype);
    MPI_Type_free(&recvtype);
}