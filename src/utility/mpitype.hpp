#ifndef UTILITY_MPITYPE_HPP
#define UTILITY_MPITYPE_HPP

#include "mpi.h"

namespace utility{
template<class value_t> inline MPI_Datatype MPI_Type() {
    std::cerr << "(mpitype.h): NULL datatype returned" << std::endl;
    exit(1);
    return 0;
}

template <> inline MPI_Datatype MPI_Type<char>() { return MPI_CHAR; }
template <> inline MPI_Datatype MPI_Type<unsigned char>() { return MPI_UNSIGNED_CHAR; }
template <> inline MPI_Datatype MPI_Type<int>() { return MPI_INT; }
template <> inline MPI_Datatype MPI_Type<long int>() { return MPI_LONG; }
template <> inline MPI_Datatype MPI_Type<unsigned int>() { return MPI_UNSIGNED; }
template <> inline MPI_Datatype MPI_Type<unsigned long int>() { return MPI_UNSIGNED_LONG; }
template <> inline MPI_Datatype MPI_Type<float>() { return MPI_FLOAT; }
template <> inline MPI_Datatype MPI_Type<double>() { return MPI_DOUBLE; }
template <> inline MPI_Datatype MPI_Type<long double>() { return MPI_LONG_DOUBLE; }
}//namespace utility

#endif
