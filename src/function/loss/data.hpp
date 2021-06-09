#ifndef FUNCTION_LOSS_DATA_HPP_
#define FUNCTION_LOSS_DATA_HPP_

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace function {
namespace loss {
template <class value_t> struct data {
    using matrix_t = std::shared_ptr<matrix::amatrix<value_t>>;
    using vector_t = std::shared_ptr<const std::vector<value_t>>;
    data() = default;
    data(const data &) = default;
    data &operator=(const data &) = default;
    data(data &&) = default;
    data &operator=(data &&) = default;

    data(matrix_t A, vector_t b) : A(std::move(A)), b(std::move(b)) {
        if (size_t(this->A->nrows()) != this->b->size())
            throw std::domain_error("data: dimension mismatch in construction");
    }
    data(matrix::dmatrix<value_t> A, std::vector<value_t> b) {
        if (size_t(A.nrows()) != b.size())
            throw std::domain_error("data: dimension mismatch in construction");
        this->A.reset(new matrix::dmatrix<value_t>(std::move(A)));
        this->b.reset(new std::vector<value_t>(std::move(b)));
    }
    data(matrix::smatrix<value_t> A, std::vector<value_t> b) {
        if (size_t(A.nrows()) != b.size())
            throw std::domain_error("data: dimension mismatch in construction");
        this->A.reset(new matrix::smatrix<value_t>(std::move(A)));
        this->b.reset(new std::vector<value_t>(std::move(b)));
    }

    int nsamples() const noexcept { return A->nrows(); }
    int nfeatures() const noexcept { return A->ncols(); }
    value_t density() const noexcept { return A->density(); }
    size_t size() const noexcept {
        return A->size() + sizeof(value_t) * b->size();
    }

    void residual(const value_t *x, value_t *r) const noexcept {
        std::copy(std::begin(*b), std::end(*b), r);
        A->mult_add('n', 1, x, -1, r);
    }
    void residual(const value_t *x, value_t *r, const int *ibegin,
                const int *iend) const noexcept {
        int k{0};
        const int *itemp{ibegin};
        while (itemp != iend)
            r[k++] = (*b)[*itemp++];
        A->mult_add('n', 1, x, -1, r, ibegin, iend);
    }

    matrix_t matrix() const noexcept { return A; }
    vector_t labels() const noexcept { return b; }

    void save(const std::string &filename) const {
        std::ofstream file(filename, std::ios_base::binary);
        A->save(file);
        file.write(reinterpret_cast<const char *>(b->data()),
                    b->size() * sizeof(value_t));
    }
    void load(const std::string &filename, bool dense) {
        std::ifstream file(filename, std::ios_base::binary);
        if (!file)
            throw std::runtime_error(filename + " could not be opened.");

        A.reset();
        b.reset();
        if (dense) {
            matrix::dmatrix<value_t> matrix_;
            matrix_.load(file);
            A.reset(new matrix::dmatrix<value_t>(std::move(matrix_)));
        } else {
            matrix::smatrix<value_t> matrix_;
            matrix_.load(file);
            A.reset(new matrix::smatrix<value_t>(std::move(matrix_)));
        }
        std::vector<value_t> labels(A->nrows());
        file.read(reinterpret_cast<char *>(&labels[0]),
                    A->nrows() * sizeof(value_t));
        b.reset(new std::vector<value_t>(std::move(labels)));
    }
    ~data() = default;
private:
    std::shared_ptr<matrix::amatrix<value_t>> A;
    std::shared_ptr<const std::vector<value_t>> b;
};
} // namespace loss
} // namespace function

#endif
