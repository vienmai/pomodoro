#ifndef MATRIX_AMATRIX_HPP_
#define MATRIX_AMATRIX_HPP_

#include <fstream>
#include <string>
#include <vector>

namespace matrix {
template <class value_t> 
struct amatrix {
    amatrix() noexcept = default;
    amatrix(const int nrows) noexcept : nrows_{nrows}, ncols_{nrows} {}
    amatrix(const int nrows, const int ncols) noexcept
        : nrows_{nrows}, ncols_{ncols} {}

    int nrows() const noexcept { return nrows_; }
    int ncols() const noexcept { return ncols_; }
    virtual value_t density() const noexcept {
        return nrows_ * ncols_ == 0 ? 0 : 1;
    }
    virtual std::size_t size() const noexcept {
        return nrows_ * ncols_ * sizeof(value_t);
    }

    virtual const value_t *data() const noexcept = 0;
    virtual value_t operator()(const int row, const int col) const = 0;
    virtual std::vector<value_t> getrow(const int row) const = 0;
    virtual std::vector<int> colindices(const int row) const = 0;

    virtual void scale_rows(const value_t *b) noexcept = 0;
    virtual void mult_add(const char trans, const value_t alpha, const value_t *x,
                          const value_t beta, value_t *y) const noexcept = 0;
    virtual void mult_add(const char trans, const value_t alpha, const value_t *x,
                          const value_t beta, value_t *y, const int *rbegin,
                          const int *rend) const noexcept = 0;

    void mult(const char trans, const value_t alpha, const value_t *x, value_t *y) 
    const noexcept {
        mult_add(trans, alpha, x, 0, y);
    }
    void mult(const char trans, const value_t alpha, const value_t *x, value_t *y,
              const int *rbegin, const int *rend) const noexcept {
        mult_add(trans, alpha, x, 0, y, rbegin, rend);
    }

    void save(const std::string &filename) const {
        std::ofstream file(filename, std::ios_base::binary);
        save(file);
    };
    virtual void save(std::ostream &os) const = 0;
    void load(const std::string &filename) {
        std::ifstream file(filename, std::ios_base::binary);
        load(file);
    };
    virtual void load(std::istream &is) = 0;

    virtual ~amatrix() = default;

    protected:
        void nrows(const int nrows) noexcept { nrows_ = nrows; }
        void ncols(const int ncols) noexcept { ncols_ = ncols; }

    private:
        int nrows_{0}, ncols_{0};
};
} 

#endif
