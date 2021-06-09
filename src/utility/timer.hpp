#ifndef UTILITY_TIMER_HPP_
#define UTILITY_TIMER_HPP_

#include <iostream>
#include <string>
#include <chrono>

namespace utility {
class timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> tstart;
public:
    timer() : tstart{std::chrono::high_resolution_clock::now()} {}
    void elapsed(const std::string& msg = "Elapsed time: ") noexcept {
        auto tend = std::chrono::high_resolution_clock::now();
        auto telapsed =
            std::chrono::duration<double, std::chrono::milliseconds::period>(tend - tstart);
        std::cout << msg << telapsed.count() << " (ms)\n";
        tstart =  std::chrono::high_resolution_clock::now();
    };
};
} // namespace utility
#endif
