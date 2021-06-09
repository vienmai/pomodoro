#ifndef TERMINATOR_ITERATION_HPP_
#define TERMINATOR_ITERATION_HPP_

namespace terminator {
template <class value_t = double> struct iteration {
  iteration(const int K) : K{K} {}

  template <class InputIt1, class InputIt2>
  bool operator()(const int k, const value_t fval, InputIt1 x_begin,
                  InputIt1 x_end, InputIt2 g_begin) const {
    return k > K;
  }

private:
  const int K;
};
} // namespace terminator

#endif
