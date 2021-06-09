#!/bin/bash
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 
export OMP_NUM_THREADS=1 
export MKL_NUM_THREADS=1 

for n in 2 4 8 16 
do
    echo "Runing with $n MPI processes."	
    OMP_NUM_THREADS=1 mpiexec -n $n ./build/examples/admm -i 100 -v 1 -p 1 -k 30
    OMP_NUM_THREADS=1 mpiexec -n $n ./build/examples/aa_admm -i 100 -v 1 -p 1 -k 30
done 

