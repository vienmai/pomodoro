#ifndef UTILITY_PRINTER_HPP_
#define UTILITY_PRINTER_HPP_

#include <iostream>
#include <vector>

namespace utility {
template <class value_t>
void printvec(std::vector<value_t> &x){
    for (auto const &val: x)
        std::cout << " " << val << " ";
    std::cout << std::endl;
}
template <class value_t>
void printvec(int begin, int end, std::vector<value_t> &x){
    for (int i = begin; i < end; i++)
        std::cout << " " << x[i] << " ";
    std::cout << std::endl;
}
template <class value_t>
void printarr(int n, value_t  *x){
    for (int i = 0; i < n; i++)
        std::cout << " " << x[i] << " ";
    std::cout << std::endl;
}
template <class value_t>
void printarr(int begin, int end, const  value_t *x){
    for (int i = begin; i < end; i++)
        std::cout << " " << x[i] << " ";
    std::cout << std::endl;
}
template <class value_t>
void printarr(int m, int n, value_t *x){
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++)
            std::cout << " " << x[i*m +j] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
}//namespace utility
#endif

