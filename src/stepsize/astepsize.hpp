#ifndef STEPSIZE_ASTEPSIZE_HPP_
#define STEPSIZE_ASTEPSIZE_HPP_

namespace stepsize {
template <class value_t> struct astepsize {
    astepsize() = default;
    bool is_linesearch() const noexcept { return false; }
    // virtual value_t operator()( ... ) noexcept = 0;
    virtual ~astepsize() = default;
};
} // namespace step

#endif
