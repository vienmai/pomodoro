#ifndef PROX_NONE_HPP_
#define PROX_NONE_HPP_

namespace prox {
template <class value_t>
struct none {
    none() = default;
    none(const none &) = default;
    none &operator=(const none &) = default;
    none(none &&) = default;
    none &operator=(none &&) = default;

    // std::vector<value_t> prox(const std::vector<value_t> &x) const noexcept { return x; }//copy
    void prox(const std::vector<value_t> &x, std::vector<value_t> &out) const noexcept { out = x; } //copy

    template <class InputIt1, class InputIt2, class OutputIt>
    OutputIt proxgrad(const value_t step, InputIt1 xbegin,
                      InputIt1 xend, InputIt2 gbegin, OutputIt obegin) const noexcept {
        std::transform(xbegin, xend, gbegin, obegin,
                       [step](const value_t xval, const value_t gval) { return xval - step * gval; });
        return obegin;
    }
protected:    
    void parameters() { }
};
} // namespace prox

#endif


