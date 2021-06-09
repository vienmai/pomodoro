#ifndef TERMINATOR_COMBINE_HPP_
#define TERMINATOR_COMBINE_HPP_

#include <cmath>
#include <iterator>
#include <type_traits>
#include <vector>

namespace terminator {
template <class value_t = double, int p = 2>
struct combine {
    combine(const int K, const value_t xtol,
            const value_t rtol, const value_t fabs, const value_t frel)
        : K{K}, xtol{xtol}, rtol{rtol}, fabs{fabs}, frel{frel} { }

    template <class InputIt1>
    bool operator()(const int k, const value_t fval, const value_t r, 
        InputIt1 x_begin, InputIt1 x_end){
        if(k > K){
            std::cout << "Terminated by iteration" << std::endl;
            return true;
        }
        const value_t fdiff = fprev - fval;
        if ((std::abs(fdiff) < fabs) | (std::abs(fdiff / (fprev + eps)) < frel)){
            std::cout << "Terminated by value" << std::endl;
            return true;
        }
        fprev = fval;
        // if ((k > 0) && (r < rtol)){
        //     std::cout << "Terminated by feasibility" << std::endl;
        //     std::cout << "k = " << k << ", res = " << r << std::endl;
        //     return true;
        // }
        const size_t d = std::distance(x_begin, x_end);
        if (xprev.size() != d){
            xprev = std::vector<value_t>(x_begin, x_end);
            return false;
        }
        if (terminate(x_begin, x_end, std::begin(xprev),
                         std::integral_constant<int, p>{})){
            std::cout << "Terminated by iterate" << std::endl;
            return true;
        }
        return false;
}

private:
    template <class InputIt, class OutputIt, int _p>
    bool terminate(InputIt x_begin, InputIt x_end, OutputIt xprev_begin,
                   std::integral_constant<int, _p>) const{
        value_t norm{0};
        while (x_begin != x_end){
            const value_t val = *x_begin++;
            const value_t diff = std::abs(*xprev_begin - val);
            norm += std::pow(diff, value_t(_p));
            *xprev_begin++ = val;
        }
        norm = std::pow(norm, value_t(1) / value_t(_p));
        return norm < xtol;
    }
    template <class InputIt, class OutputIt>
    bool terminate(InputIt x_begin, InputIt x_end, OutputIt xprev_begin,
                   std::integral_constant<int, -1>) const{
        value_t max{0};
        while (x_begin != x_end){
            const value_t val = *x_begin++;
            const value_t diff = std::abs(*xprev_begin - val);
            if (diff > max)
                max = diff;
            *xprev_begin++ = val;
        }
        return max < xtol;
    }
    template <class InputIt, class OutputIt>
    bool terminate(InputIt x_begin, InputIt x_end, OutputIt xprev_begin,
                   std::integral_constant<int, 1>) const{
        value_t norm{0};
        while (x_begin != x_end){
            const value_t val = *x_begin++;
            norm += std::abs(*xprev_begin - val);
            *xprev_begin++ = val;
        }
        return norm < xtol;
    }
    template <class InputIt, class OutputIt>
    bool terminate(InputIt x_begin, InputIt x_end, OutputIt xprev_begin,
                   std::integral_constant<int, 2>) const{
        value_t norm{0};
        while (x_begin != x_end){
            const value_t val = *x_begin++;
            const value_t diff = *xprev_begin - val;
            norm += diff * diff;
            *xprev_begin++ = val;
        }
        return std::sqrt(norm) < xtol;
    }
    
    const int K;
    std::vector<value_t> xprev;
    value_t fprev{1E10};
    const value_t xtol, rtol, fabs, frel;
    const value_t eps = std::numeric_limits<value_t>::epsilon();
};
} // namespace terminator
#endif
