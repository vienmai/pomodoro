# Pomodoro: Progressive Decomposition Methods with Acceleration

`Pomodoro` is an MPI C++ implementation of an abstract acceleration framework for solving distributed optimization problems. The algorithms accelerate the `Progressive decomposition` method in [(Rockafellar, SVA-2018)](https://sites.math.washington.edu/~rtr/papers/rtr252-Decoupling.pdf) by using several different optimization acceleration techniques. In particular, it uses the power of Anderson acceleration to obtain fast convergence and scalability to multiple workers (see, e.g., our paper [(Mai and Johansson, ICML-2020)](https://arxiv.org/pdf/1910.08590.pdf) for background on Anderson acceleration for proximal convex optimization).

Apart from a compiler that supports C++17 features, we also have the following requirements:
1. `MPI` 
2. `CMAKE`
3. `BLAS` and `LAPACK`

# Problem

We consider optimization on the form:
<!-- $$
\underset{x\in \mathbb{R}^n }{\text{minimize}} \quad 
    F(x) = \sum_{i=1}^{m}f_i(x) + g(x) \quad 
 \text{subject to}  \quad x\in S,
$$ -->
![Alt text](/examples/figures/equation.png?raw=true "Optional Title")
where $f_i(x)$ models the individual loss of agent $i$, g(x) is a regulerizer, and the linear subspace $S$ represents coupling constraints between agents. The component functions $f_i(x)$ and the decision vector are distributed among the $m$  agents.

## Example: Logistic Regression

The following listing shows how to use the solver function in `pomodoro` to solve a logistic regression problem on the `real-sim` dataset that contains 72309 samples and 20958 features. We use L-BFGS as the subproblem solver. More examples can be found in the `examples/` directory.

```cpp
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

int main(int argc, char *argv[]){
    // Setting up MPI processes
    int nprocs, rank;
    MPI_Comm COMM;
    MPI_Init(NULL, NULL);
    COMM = MPI_COMM_WORLD;
    MPI_Comm_size(COMM, &nprocs);
    MPI_Comm_rank(COMM, &rank);

    // Parameters
    int m, dataset_choice{8};
    optimizer::admm_params<double> params;
    optimizer::lbfgs_params<double> subsolver_params;
    getargv(argc, argv, dataset_choice, params, subsolver_params);
    params.nprocs = nprocs;

    // Load and distribute the data set to `nprocs` workers
    auto dataset = datasets.at(dataset_choice);
    auto dataloc = getdata<double>(dataset, m, rank, nprocs, COMM);
    
    // Specify a local loss function
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

    // Pick a subprolem solver
    optimizer::lbfgs<double, stepsize::linesearch> subsolver(subsolver_params);
    logger::valfeas<double> logger;
    terminator::combine<double> terminator(params.max_iters, 1E-10, 1E-10, 1E-10, 1E-10);

    // Initialize a global decision vector        
    int n = dataloc.nfeatures();
    std::vector<double> x(n, 1 / (double) n);
    MPI_Bcast(&x[0], x.size(), MPI_DOUBLE, 0, COMM);
    alg.initialize(x);

    // Solve the problem
    alg.solve(subsolver, lossloc, logger, terminator, COMM, rank);

    // Store the output
    if (rank == 0) {
      [[maybe_unused]] auto &[dataname, dims] = dataset;
      logger.csv("examples/output/" + dataname + solver
          + "_nprocs" + std::to_string(nprocs)
          + "_rho" + std::to_string(params.rho) + ".csv");
    }

    // Finalize MPI processes
    MPI_Finalize();
    return 0;
}
```

# Building and running
 For example, to compile the methods `ADMM` and `ANDERSON` listed above, we add the following lines to CMakeLists.txt:

```cmake
add_executable(admm admm.cpp)
target_compile_features(admm PRIVATE cxx_std_17)
target_compile_options(admm PRIVATE -Wall -Wpedantic -Wno-vla-extension -O2)
target_compile_definitions(admm PUBLIC ADMM)
target_link_libraries(admm lib::pomodoro)
```

```cmake
add_executable(anderson admm.cpp)
target_compile_features(anderson PRIVATE cxx_std_17)
target_compile_options(anderson PRIVATE -Wall -Wpedantic -Wno-vla-extension -O2)
target_compile_definitions(anderson PUBLIC ANDERSON)
target_link_libraries(anderson lib::pomodoro)
```
We are ready to call CMake and get our build system:

```bash
cmake -S. -Bbuild
```
And finally build our executable:
```bash
cmake --build build
```
To execute the two algorithms with 16 MPI processes, 50 iterations, and 30 iterations of L-BFGS, we run: 
```bash
OMP_NUM_THREADS=1 mpiexec -n 16 ./build/examples/admm -i 50 -k 30

OMP_NUM_THREADS=1 mpiexec -n 16 ./build/examples/anderson -i 50 -k 30
```


# Plotting
Having run the algorithms, we can now load the outputs and plot the loss values or constraint residuals with respect to iteration counts and wall-clock times.

```python
dataset = "real-sim"
optval  = optvals[dataset]
algos = ["admm", "anderson"]
labels = ["admm", "anderson"]
colors = ['C0', 'C3']
markers = ['', '.']

nprocs = [2, 4, 8, 16];
for idx in range(len(algos)):
    for proc_idx in range(len(nprocs)):
        k = []; t = []; f = []; feas = []
        with open("output/" + dataset + "_" + algos[idx] + "_nprocs" + str(nprocs[proc_idx]) + "_rho1.000000"".csv") as csvfile:
            csvReader = csv.reader(csvfile, delimiter=",")
            next(csvReader) 
            for row in csvReader:
                k.append(int(row[0]))
                t.append(float(row[1]) / 60)
                f.append(float(row[2]))
                feas.append(float(row[3])) # Feasibility
        
        plt.plot(k, [f - optval for f in f], color=colors[idx], marker=markers[proc_idx], label=labels[idx]+"-"+str(nprocs[proc_idx]))
        # plt.plot(t, [f - optval for f in f], color=colors[idx], label=labels[idx])
        # plt.plot(k, [fe for fe in feas], color=colors[idx], label=labels[idx])
plt.yscale('log')
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Objective value")
```
![Alt text](/examples/figures/example.png?raw=true "Optional Title")

See the `Jupyter` notebook under the `examples/` directory for more detail.
