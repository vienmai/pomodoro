#ifndef ACCELERATOR_NONE_HPP_
#define ACCELERATOR_NONE_HPP_

namespace accelerator {
template <class value_t>
struct none {
    none() = default;

    template <class InputIt1, class InputIt2, class OutputIt>
    OutputIt accelerate(const int k, const value_t fval, InputIt1 xbegin,
                     InputIt1 xend, InputIt2 gbegin, OutputIt obegin) const noexcept {
       return obegin;
    }

    template <class Comm, class InputIt1, class InputIt2, class OutputIt1, class OutputIt2>
    void accelerate(const Comm &COMM, const int rank, const int k, InputIt1 xbegin,
                    InputIt1 xend, InputIt2 ybegin, OutputIt1 ubegin, OutputIt2 vbegin) const noexcept {}

protected:
    void parameter() {}
    void initialize(const std::vector<value_t> &x0) {}
};
} // namespace accelerator

#endif


