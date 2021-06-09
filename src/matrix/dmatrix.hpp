#ifndef MATRIX_DMATRIX_HPP_
#define MATRIX_DMATRIX_HPP_

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "amatrix.hpp"
#include "../utility/random.hpp"

namespace matrix {
template <class value_t> struct smatrix;

template <class value_t>
struct dmatrix : public amatrix<value_t> {
    dmatrix() noexcept(noexcept(std::vector<value_t>())) = default;
    dmatrix(const dmatrix &) = delete;
    dmatrix &operator=(const dmatrix &) = delete;
    dmatrix(dmatrix &&) = default;
    dmatrix &operator=(dmatrix &&) = default;
    ~dmatrix() = default;
    
    const value_t *data() const noexcept override { return &values_[0]; }

    dmatrix(const int nrows)
        : amatrix<value_t>(nrows), values_(nrows * nrows) {}
    dmatrix(const int nrows, const int ncols)
        : amatrix<value_t>(nrows, ncols), values_(nrows * ncols) {}
    dmatrix(const int nrows, const int ncols, std::vector<value_t> values)
        : amatrix<value_t>(nrows, ncols) {
        if (values.size() != std::size_t(nrows) * ncols)
            throw std::domain_error("dmatrix: dimension mismatch in construction");
        values_ = std::move(values);
    }

    value_t operator()(const int row, const int col) const override {
        const int nrows_ = amatrix<value_t>::nrows();
        const int ncols_ = amatrix<value_t>::ncols();
        if (row >= nrows_ || col >= ncols_)
            throw std::range_error("row or column out of range.");
        return values_[col * nrows_ + row];
    }
    value_t &operator()(const int row, const int col){
        const int nrows_ = amatrix<value_t>::nrows();
        const int ncols_ = amatrix<value_t>::ncols();
        if (row >= nrows_ || col >= ncols_)
            throw std::range_error("row or column out of range.");

        return this->values_[col * nrows_ + row];
    }
    std::vector<value_t> getrow(const int row) const override {
        const int nrows_ = amatrix<value_t>::nrows();
        const int ncols_ = amatrix<value_t>::ncols();
        if (row >= nrows_)
            throw std::range_error("row out of range.");
        std::vector<value_t> rowvec(ncols_);
        for (int col = 0; col < ncols_; col++)
            rowvec[col] = values_[col * nrows_ + row];
        return rowvec;
    }

    std::vector<int> colindices(const int row) const override {
        const int nrows_ = amatrix<value_t>::nrows();
        const int ncols_ = amatrix<value_t>::ncols();
        if (row >= nrows_)
            throw std::range_error("row out of range.");
        std::vector<int> indices(ncols_);
        int idx{0};
        std::transform(std::begin(indices), std::end(indices), std::begin(indices),
                    [&](const int) { return idx++; });
        return indices;
    }
    void scale_rows(const value_t *b) noexcept override {
        const int nrows_ = amatrix<value_t>::nrows();
        const int ncols_ = amatrix<value_t>::ncols();
        for (int col = 0; col < ncols_; col++) {
            for (int row = 0; row < nrows_; row++)
                this->values_[col * nrows_ + row] *= *(b + row);
        }
    }
    void matrix_mult_add(const value_t alpha, const value_t beta, value_t *c) {
        // Compute: C = alpha * A^T A + beta * C
        // where A: nrows x ncols 
        const int nrows = amatrix<value_t>::nrows();
        const int ncols = amatrix<value_t>::ncols();
        algebra::matrix::blas<value_t>::gemm('t', 'n', ncols, ncols, nrows, alpha, &values_[0],
                                             nrows, &values_[0], nrows, beta, c, ncols);
        }
    
    void mult_add(const char trans, const value_t alpha, const value_t *x,
                    const value_t beta, value_t *y) const noexcept override {
        const int nrows = amatrix<value_t>::nrows();
        const int ncols = amatrix<value_t>::ncols();
        algebra::matrix::blas<value_t>::gemv(trans, nrows, ncols, alpha,
                                            &values_[0], nrows, x, 1, beta, y, 1);
    }
    void mult_add(const char trans, const value_t alpha, const value_t *x,
                    const value_t beta, value_t *y, const int *rbegin,
                    const int *rend) const noexcept override {
        const int nrows = amatrix<value_t>::nrows();
        const int ncols = amatrix<value_t>::ncols();
        const bool notrans = (trans == 'N') | (trans == 'n');
        value_t *yend = y + (notrans ? std::distance(rbegin, rend) : size_t(ncols));
        std::transform(y, yend, y, [=](const value_t val) { return beta * val; });
        if (notrans)  
            while (rbegin != rend) {
                const int row = *rbegin++;
                algebra::matrix::blas<value_t>::gemv(
                    trans, 1, ncols, alpha, &values_[row], nrows, x, 1, 1, y++, 1);
            }
        else
            while (rbegin != rend) {
                const int row = *rbegin++;
                algebra::matrix::blas<value_t>::gemv(
                    trans, 1, ncols, alpha, &values_[row], nrows, x++, 1, 1, y, 1);
            }
    }

    void save(std::ostream &os) const override {
        const int nrows_ = amatrix<value_t>::nrows();
        const int ncols_ = amatrix<value_t>::ncols();
        os.write(reinterpret_cast<const char *>(&nrows_), sizeof(int));
        os.write(reinterpret_cast<const char *>(&ncols_), sizeof(int));
        os.write(reinterpret_cast<const char *>(&values_[0]),
                std::size_t(nrows_) * ncols_ * sizeof(value_t));
    }
    void load(std::istream &is) override {
        int nrows_, ncols_;
        is.read(reinterpret_cast<char *>(&nrows_), sizeof(int));
        is.read(reinterpret_cast<char *>(&ncols_), sizeof(int));
        amatrix<value_t>::nrows(nrows_);
        amatrix<value_t>::ncols(ncols_);
        values_ = std::vector<value_t>(std::size_t(nrows_) * ncols_);
        is.read(reinterpret_cast<char *>(&values_[0]),
                std::size_t(nrows_) * ncols_ * sizeof(value_t));
    }

    template <class U>
    friend std::ostream &operator<<(std::ostream &stream, const matrix::dmatrix<U> &A){
        A.print(stream);
        return stream;
    }

    smatrix<value_t>
    sparse(const value_t eps = std::numeric_limits<value_t>::min()) const {
        if (eps < 0)
            throw std::domain_error("dmatrix::sparse: negative eps is not allowed");
        const int nrows = amatrix<value_t>::nrows();
        const int ncols = amatrix<value_t>::ncols();
        std::vector<int> row_ptr{0}, cols;
        std::vector<value_t> values;
        for (int row = 0; row < nrows; row++) 
        {
            for (int col = 0; col < ncols; col++) {
                const value_t val = values_[col * nrows + row];
                if ((val > eps) | (val < -eps)) {
                values.push_back(val);
                cols.push_back(col);
                }
            }
            row_ptr.push_back(cols.size());
        }
        return smatrix<value_t>(nrows, ncols, std::move(row_ptr),
                                        std::move(cols), std::move(values));
    }

private:
    std::vector<value_t> values_;
    
    void print(std::ostream &stream) const {
        const int nrows_ = amatrix<value_t>::nrows();
        const int ncols_ = amatrix<value_t>::ncols();
        for (int i = 0; i < nrows_; i++){
            for (int j = 0; j < ncols_; j++)
                stream << " " << (*this)(i, j) << " ";
            stream << std::endl;
        }
    }
};

template <class value_t>
inline dmatrix<value_t> randn(const size_t m, const size_t n) {
    return dmatrix<value_t>(m, n, utility::randn<value_t>(m * n));
}
template <class value_t>
inline dmatrix<value_t> rand(const size_t m, const size_t n) {
    return dmatrix<value_t>(m, n, utility::rand<value_t>(m * n));
}
template <class value_t>
inline dmatrix<value_t> ones(const size_t m, const size_t n) {
    std::vector<value_t> values(m * n, 1);
    return dmatrix<value_t>(m, n, std::move(values));
}
} // namespace matrix 
#endif
