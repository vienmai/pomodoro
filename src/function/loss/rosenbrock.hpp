#ifndef FUNCTION_ROSENBROCK_HPP_
#define FUNCTION_ROSENBROCK_HPP_

namespace function {
namespace loss {
struct rosenbrock {
    rosenbrock() = default;
    rosenbrock(const double a, const double b) : a(a), b(b) {}

    double operator()(const double *xvec) const noexcept {
        const double x = xvec[0];
        const double y = xvec[1];
        return (a - x) * (a - x) + b * (y - x * x) * (y - x * x);
    }
    double operator()(const double *xvec, double *g) const noexcept {
        const double x = xvec[0];
        const double y = xvec[1];
        g[0] = -2 * (a - x) - 4 * b * (y - x * x) * x;
        g[1] = 2 * b * (y - x * x);
        return (a - x) * (a - x) + b * (y - x * x) * (y - x * x);
    }
    void grad(const double *xvec, double *g) const noexcept {
        const double x = xvec[0];
        const double y = xvec[1];
        g[0] = -2 * (a - x) - 4 * b * (y - x * x) * x;
        g[1] = 2 * b * (y - x * x);
    }
private:
    const double a{1}, b{100};    
};

} // namespace function
} // namespace loss
#endif