#ifndef FUNCTION_LOSS_ALOSS_HPP_
#define FUNCTION_LOSS_ALOSS_HPP_

#include "data.hpp"

namespace function {
namespace loss {
template <class value_t> struct aloss {
    using data_t = function::loss::data<value_t>;
    using matrix_t = typename data_t::matrix_t;
    using vector_t = typename data_t::vector_t;

    aloss() = default;
    aloss(data_t data) : data_(std::move(data)) {}

    void data(data_t data) { data_ = std::move(data); }
    data_t data() const noexcept { return data_; }

    int nsamples() const noexcept { return data_.nsamples(); }
    int nfeatures() const noexcept { return data_.nfeatures(); }

    matrix_t matrix() const noexcept { return data_.matrix(); }
    vector_t labels() const noexcept { return data_.labels(); }

    virtual value_t operator()(const value_t *x) noexcept = 0;
    virtual value_t operator()(const value_t *x, value_t *g) noexcept = 0;
    virtual value_t operator()(const value_t *x, value_t *g, 
                               const int *ib, const int *ie) noexcept = 0;

    virtual void grad(const value_t *x, value_t *g) noexcept = 0;

    virtual ~aloss() = default;

protected:
    data_t data_;
};
} // namespace loss
} // namespace function

#endif



   