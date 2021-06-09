#ifndef UTILITY_NULL_HPP_
#define UTILITY_NULL_HPP_

namespace utility {
namespace detail {
struct null {
  template <class... Args> void operator()(Args &&...) {}
};
} // namespace detail
} // namespace utility

#endif
