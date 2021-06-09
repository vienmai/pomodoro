#ifndef UTILITY_RANDOM_HPP_
#define UTILITY_RANDOM_HPP_

#include <iterator>
#include <random>
#include <utility>
#include <algorithm>

namespace utility {
template <class value_t, template <class> class distribution_t>
struct random : private distribution_t<value_t> {
    using param_type = typename distribution_t<value_t>::param_type;
    using result_type = typename distribution_t<value_t>::result_type;
    using vector_t = std::vector<value_t>;

    random() = default;
    random(distribution_t<value_t> dist)
        : distribution_t<value_t>(std::move(dist)) {}
    random(const random &s) : distribution_t<value_t>(s) {}
    random &operator=(const random &) = default;
    random(random &&) = default;
    random &operator=(random &&) = default;
    ~random() = default;

    template <class gen_t> static void setgen(gen_t gen){ gen(std::move(gen)); }
    template <class... Ts> static void seed(const Ts &... seed) { gen.seed(seed...); }
    template <class... Ts> void parameters(const Ts &... params) {
        distribution_t<value_t>::param(param_type(params...));
    }
    void parameters(const param_type &params) { distribution_t<value_t>::param(params); }
    param_type parameters() const { return distribution_t<value_t>::param(); }

    vector_t operator()(const size_t n) noexcept {
        vector_t values(n);
        std::generate(std::begin(values), std::end(values),
                      [&, this]() { return distribution_t<value_t>::operator()(this->gen); });
        return values;
    }
    template <class OutputIt>
    OutputIt operator()(OutputIt xbegin, OutputIt xend) noexcept {
        vector_t values = this->operator()(std::distance(xbegin, xend));
        for (auto &val : values)
            *xbegin++ = val;
        return xbegin;
    }

private:
    inline static std::mt19937 gen{std::random_device{}()}; // C++17
};

template <class value_t = double>
using normal = random<value_t, std::normal_distribution>;
template <class value_t = double>
using uniform = random<value_t, std::uniform_real_distribution>;
template <class value_t = int>
using uniform_discrete = random<value_t, std::uniform_int_distribution>;

template <class value_t>
inline std::vector<value_t> randn(const size_t n) { 
    return normal<value_t>()(n);
}
template <class value_t>
inline std::vector<value_t> rand(const size_t n) { 
    return uniform<value_t>()(n);
}
} // namespace utility
#endif
