#ifndef UTILITY_GETARGV_HPP_
#define UTILITY_GETARGV_HPP_

#include <iostream>
#include <map>
#include <array>
#include <unistd.h>

using namespace optimizer;

namespace utility {
template<class value_t>
void getargv(int argc, char **argv, params<value_t> &params, stepsize_params<value_t> &stepsize_params){
    int c;
    while ((c = getopt(argc, argv, "i:v:e:p:g:s:l::a:r:x:h")) != -1)
        switch (c){
        case 'i':
            params.max_iters = atoi(optarg);
            break;
        case 'v':
            params.verbose = atoi(optarg);
            break;
        case 'e':
            params.epsilon = atoi(optarg);
            break;
        case 'p':
            params.print_every = atoi(optarg);
            break;
        case 'g':
            params.log_every = atoi(optarg);
            break;
        case 's':
            stepsize_params.init_step = atoi(optarg);
            break;
        case 'l':
            stepsize_params.is_ls = atoi(optarg);
            break;
        case 'a':
            stepsize_params.ls_adapt = atoi(optarg);
            break;
        case 'r':
            stepsize_params.ls_ratio = atoi(optarg);
            break;
        case 'x':
            stepsize_params.max_step = atoi(optarg);
            break;
        case 'h':
            printf("Options:\n-i \tSolver choice\n"
                   "-i \tIteration number\n"
                   "-v \tVerbose or not\n"
                   "-e \tEpsilon\n"
                   "-p \tPrint every\n"
                   "-s \tStepsize\n"
                   "-l \tLinesearch\n"
                   "-a \tAdaptive linesearch\n"
                   "-r \tLinesearch ratio\n"
                   "-x \tMaximum stepsize\n");
            exit(1);
        case '?':
            break;
        }
}
template<class value_t>
void getargv(int argc, char **argv, aa_params<value_t> &params){
    int c;
    while ((c = getopt(argc, argv, "i:v:e:p:g:m:s:r:h")) != -1)
        switch (c){
        case 'i':
            params.max_iters = atoi(optarg);
            break;
        case 'v':
            params.verbose = atoi(optarg);
            break;
        case 'e':
            params.epsilon = atoi(optarg);
            break;
        case 'p':
            params.print_every = atoi(optarg);
            break;
        case 'g':
            params.log_every = atoi(optarg);
            break;
        case 'm':
            params.memory = atoi(optarg);
            break;
        case 'r':
            params.reg = atoi(optarg);
            break;
        case 'h':
            printf("Options:\n-i \tSolver choice\n"
                   "-i \tIteration number\n"
                   "-v \tVerbose or not\n"
                   "-e \tEpsilon\n"
                   "-p \tPrint every\n"
                   "-m \tMemory\n"
                   "-r \tRegularization\n");
            exit(1);
        case '?':
            break;
        }
}

template<class value_t>
void getargv(int argc, char **argv, lbfgs_params<value_t> &params){
    int c;
    while ((c = getopt(argc, argv, "i:v:e:p:g:m:s:l::a:r:o:x:c:y:h")) != -1)
        switch (c){
        case 'i':
            params.max_iters = atoi(optarg);
            break;
        case 'v':
            params.verbose = atoi(optarg);
            break;
        case 'e':
            params.epsilon = atoi(optarg);
            break;
        case 'p':
            params.print_every = atoi(optarg);
            break;
        case 'g':
            params.log_every = atoi(optarg);
            break;
        case 'm':
            params.memory = atoi(optarg);
            break;
        case 's':
            params.init_step = atoi(optarg);
            break;
        case 'l':
            params.is_ls = atoi(optarg);
            break;
        case 'a':
            params.ls_adapt = atoi(optarg);
            break;
        case 'r':
            params.ls_ratio = atoi(optarg);
            break;
        case 'o':
            params.ls_rho = atoi(optarg);
            break;
        case 'c':
            params.ls_dec = atoi(optarg);
            break;
        case 'x':
            params.max_step = atoi(optarg);
            break;
        case 'y':
            params.ls_maxiter = atoi(optarg);
            break;
        case 'h':
            printf("Options:\n-i \tSolver choice\n"
                   "-i \tIteration number\n"
                   "-v \tVerbose or not\n"
                   "-e \tEpsilon\n"
                   "-p \tPrint every\n"
                   "-m \tMemory\n"
                   "-s \tStepsize\n"
                   "-l \tLinesearch\n"
                   "-a \tAdaptive linesearch\n"
                   "-r \tLinesearch ratio\n"
                   "-o \tLinesearch parameter rho\n"
                   "-x \tMaximum stepsize\n"
                   "-y \tMaximum number of linesearch\n");
            exit(1);
        case '?':
            break;
        }
}
template<class value_t>
void getargv(int argc, char **argv, int &dataset_choice, admm_params<value_t> &params, lbfgs_params<value_t> &lbfgs_params){
    int c;
    while ((c = getopt(argc, argv, "j:i:v:e:p:r:m:s:l:a:d:o:c:x:y:z:k:h")) != -1)
        switch (c){
        case 'j':
            dataset_choice = atoi(optarg);
            break;
        case 'i':
            params.max_iters = atoi(optarg);
            break;
        case 'v':
            params.verbose = atoi(optarg);
            break;
        case 'p':
            params.print_every = atoi(optarg);
            break;
        case 'r':
            params.rho = atoi(optarg);
            break;
        case 'e':
            params.elic = atoi(optarg);
            break;
        case 'm':
            lbfgs_params.memory = atoi(optarg);
            break;
        case 's':
            lbfgs_params.init_step = atoi(optarg);
            break;
        case 'l':
            lbfgs_params.is_ls = atoi(optarg);
            break;
        case 'a':
            lbfgs_params.ls_adapt = atoi(optarg);
            break;
        case 'd':
            lbfgs_params.ls_ratio = atoi(optarg);
            break;
        case 'o':
            lbfgs_params.ls_rho = atoi(optarg);
            break;
        case 'c':
            lbfgs_params.ls_dec = atoi(optarg);
            break;
        case 'x':
            lbfgs_params.max_step = atoi(optarg);
            break;
        case 'y':
            lbfgs_params.ls_maxiter = atoi(optarg);
            break; 
        case 'z':
            lbfgs_params.verbose = atoi(optarg);
            break;
        case 'k':
            lbfgs_params.max_iters = atoi(optarg);
            break;
        case 'h':
           printf("ADMM-LBFGS Options: \n-i \tIteration number\n"
                                        "-v \tVerbose or not\n"                                       
                                        "-p \tPrint every\n"
                                        "-r \tParameter rho\n"
                                        "-e \tElicitation parameter\n"
                                        "-m \tlbfgs memory\n"
                                        "-s \tlbfgs init stepsize\n"
                                        "-a \tlbfgs adapt linesearch?\n"
                                        "-l \tlbfgs is linesearch?\n"
                                        "-d \tlbfgs linesearch increase ratio\n"
                                        "-o \tlbfgs linesearch rho parameter\n"
                                        "-x \tlbfgs linesearch maximum stepsize\n"
                                        "-y \tlbfgs maximum number of linesearch steps\n"
                                        "-z \tlbfgs verbose?\n"
                                        "-k \tlbfgs maximum number of iterarions\n");
            exit(1);
        case '?':
            break;
        }
}

}//namespace utility
#endif
