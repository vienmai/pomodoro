#ifndef MATRIX_SMATRIX_HPP_
#define MATRIX_SMATRIX_HPP_

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "amatrix.hpp"

namespace matrix {
template <class value_t> struct dmatrix;

template <class value_t>
struct smatrix : public amatrix<value_t> {
  smatrix() noexcept(noexcept(std::vector<value_t>())) = default;
  smatrix(const smatrix &) = delete;
  smatrix &operator=(const smatrix &) = delete;
  smatrix(smatrix &&) = default;
  smatrix &operator=(smatrix &&) = default;
  ~smatrix() = default;

  smatrix(const int nrows)
      : amatrix<value_t>(nrows), row_ptr_(nrows + 1) {}
  smatrix(const int nrows, const int ncols)
      : amatrix<value_t>(nrows, ncols), row_ptr_(nrows + 1) {}

  smatrix(const int nrows, const int ncols,
          std::vector<int> row_ptr, std::vector<int> cols,
          std::vector<value_t> values)
      : amatrix<value_t>(nrows, ncols) {
    if ((std::size_t(nrows) != row_ptr.size() - 1) |
        (cols.size() != values.size()))
      throw std::domain_error("smatrix: dimension mismatch in construction");
    auto res = std::max_element(std::begin(cols), std::end(cols));
    if (res != std::end(cols) && *res >= ncols)
      throw std::domain_error("smatrix: dimension mismatch in construction");
    if (values.size() > 0 && std::size_t(row_ptr.back()) != values.size())
      throw std::domain_error("smatrix: dimension mismatch in construction");
    row_ptr_ = std::move(row_ptr);
    cols_ = std::move(cols);
    values_ = std::move(values);
  }
  const value_t *data() const noexcept override { return &values_[0]; }
  
  value_t density() const noexcept override {
    const int nrows = amatrix<value_t>::nrows();
    const int ncols = amatrix<value_t>::ncols();
    return ncols * nrows == 0 ? 0 : value_t(values_.size()) / nrows / ncols;
  }
  std::size_t size() const noexcept override {
    return (row_ptr_.size() + cols_.size()) * sizeof(int) +
           values_.size() * sizeof(value_t);
  }

  value_t operator()(const int row, const int col) const override {
    const int nrows_ = amatrix<value_t>::nrows();
    const int ncols_ = amatrix<value_t>::ncols();
    if (row >= nrows_ && col >= ncols_)
      throw std::range_error("row or column out of range.");
    for (int colidx = row_ptr_[row]; colidx < row_ptr_[row + 1]; colidx++)
      if (cols_[colidx] == col)
        return values_[colidx];
    return value_t{0};
  }
  std::vector<value_t> getrow(const int row) const override {
    const int nrows_ = amatrix<value_t>::nrows();
    if (row >= nrows_)
      throw std::range_error("row out of range.");
    const int colstart = row_ptr_[row];
    const int colend = row_ptr_[row + 1];
    std::vector<value_t> rowvec(colend - colstart);
    int idx{0};
    for (int col = colstart; col < colend; col++)
      rowvec[idx++] = values_[col];
    return rowvec;
  }
  std::vector<int> colindices(const int row) const override {
    const int nrows_ = amatrix<value_t>::nrows();
    if (row >= nrows_)
      throw std::range_error("row out of range.");
    const int colstart = row_ptr_[row];
    const int colend = row_ptr_[row + 1];
    std::vector<int> indices(colend - colstart);
    int idx{0};
    for (int col = colstart; col < colend; col++)
      indices[idx++] = cols_[col];
    return indices;
  }
  void scale_rows(const value_t *b) noexcept override {};
  void mult_add(const char trans, const value_t alpha, const value_t *x,
                const value_t beta, value_t *y) const noexcept override {
    kernel(trans, alpha, x, beta, y, nullptr, nullptr);
  }
  void mult_add(const char trans, const value_t alpha, const value_t *x,
                const value_t beta, value_t *y, const int *rbegin,
                const int *rend) const noexcept override {
    kernel(trans, alpha, x, beta, y, rbegin, rend);
  }

  void save(std::ostream &os) const override {
    const int nrows_ = amatrix<value_t>::nrows();
    const int ncols_ = amatrix<value_t>::ncols();
    const std::size_t nnz_ = values_.size();
    os.write(reinterpret_cast<const char *>(&nrows_), sizeof(int));
    os.write(reinterpret_cast<const char *>(&ncols_), sizeof(int));
    os.write(reinterpret_cast<const char *>(&nnz_), sizeof(std::size_t));
    os.write(reinterpret_cast<const char *>(&row_ptr_[0]),
             row_ptr_.size() * sizeof(int));
    os.write(reinterpret_cast<const char *>(&cols_[0]), nnz_ * sizeof(int));
    os.write(reinterpret_cast<const char *>(&values_[0]),
             nnz_ * sizeof(value_t));
  }
  void load(std::istream &is) override {
    int nrows_, ncols_;
    std::size_t nnz_;
    is.read(reinterpret_cast<char *>(&nrows_), sizeof(int));
    is.read(reinterpret_cast<char *>(&ncols_), sizeof(int));
    is.read(reinterpret_cast<char *>(&nnz_), sizeof(std::size_t));
    amatrix<value_t>::nrows(nrows_);
    amatrix<value_t>::ncols(ncols_);
    row_ptr_ = std::vector<int>(std::size_t(nrows_) + 1);
    cols_ = std::vector<int>(nnz_);
    values_ = std::vector<value_t>(nnz_);
    is.read(reinterpret_cast<char *>(&row_ptr_[0]),
            row_ptr_.size() * sizeof(int));
    is.read(reinterpret_cast<char *>(&cols_[0]), nnz_ * sizeof(int));
    is.read(reinterpret_cast<char *>(&values_[0]), nnz_ * sizeof(value_t));
  }

  dmatrix<value_t> dense() const {
    const int nrows = amatrix<value_t>::nrows();
    const int ncols = amatrix<value_t>::ncols();
    std::vector<value_t> values(nrows * ncols);
    for (int row = 0; row < nrows; row++) {
      const int nnz = row_ptr_[row + 1] - row_ptr_[row];
      int colstart = row_ptr_[row];
      for (int colidx = colstart; colidx < colstart + nnz; colidx++)
        values[cols_[colidx] * nrows + row] = values_[colidx];
    }
    return dmatrix<value_t>(nrows, ncols, std::move(values));
  }

private:
  void kernel(const char trans, const value_t alpha, const value_t *x,
              const value_t beta, value_t *y, const int *rbegin,
              const int *rend) const noexcept {
    const int nrows = amatrix<value_t>::nrows();
    const int ncols = amatrix<value_t>::ncols();
    if ((trans == 'n') | (trans == 'N')) {
      if ((rbegin == nullptr) | (rend == nullptr))
        for (int row = 0; row < nrows; row++)
          notrans_f(alpha, x, row, beta, y++);
      else
        while (rbegin != rend)
          notrans_f(alpha, x, *rbegin++, beta, y++);
    } else {
      for (int col = 0; col < ncols; col++)
        y[col] *= beta;
      if ((rbegin == nullptr) | (rend == nullptr))
        for (int row = 0; row < nrows; row++)
          trans_f(alpha, x++, row, y);
      else
        while (rbegin != rend)
          trans_f(alpha, x++, *rbegin++, y);
    }
  }

  void notrans_f(const value_t alpha, const value_t *x, const int row,
                 const value_t beta, value_t *y) const noexcept {
    const int nnz = row_ptr_[row + 1] - row_ptr_[row];
    int colstart = row_ptr_[row];
    *y *= beta;
    for (int colidx = colstart; colidx < colstart + nnz; colidx++)
      *y += alpha * values_[colidx] * x[cols_[colidx]];
  }

  void trans_f(const value_t alpha, const value_t *x, const int row,
               value_t *y) const noexcept {
    const int nnz = row_ptr_[row + 1] - row_ptr_[row];
    int colstart = row_ptr_[row];
    for (int colidx = colstart; colidx < colstart + nnz; colidx++)
      y[cols_[colidx]] += alpha * values_[colidx] * (*x);
  }

  std::vector<int> row_ptr_, cols_;
  std::vector<value_t> values_;
};
} // namespace matrix

#endif
