#ifndef UTILITY_READER_HPP_
#define UTILITY_READER_HPP_

#include <fstream>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

namespace utility {
template <class value_t> struct reader {
    static function::loss::data<value_t>
    svm(const std::vector<std::string> &filenames) {
        char delim;
        int colidxmax{0}, colidx;
        value_t label, value;
        std::vector<value_t> labels, values;
        std::vector<int> row_ptr{0}, cols;

        for (const auto &file : filenames) {
            std::string line;
            std::ifstream input{file};
            if (input) {
                while (std::getline(input, line)) {
                    std::stringstream ss{line};
                    ss >> label;
                    labels.push_back(label);
                    while (ss >> colidx >> delim >> value) {
                        if (colidx > colidxmax)
                            colidxmax = colidx;
                        cols.push_back(colidx - 1);
                        values.push_back(value);
                    }
                    row_ptr.push_back(cols.size());
                }
            }
        }
        auto A = std::make_shared<matrix::smatrix<value_t>>(
                    row_ptr.size() - 1, colidxmax,
                    std::move(row_ptr), std::move(cols), std::move(values));
        auto b = std::make_shared<const std::vector<value_t>>(std::move(labels));
        return function::loss::data<value_t>(A, b);
    }
    static function::loss::data<value_t>
    svm(const std::vector<std::string> &filenames, const int nsamples,
        const int nfeatures){
        char delim;
        int rowidx{0}, colidx;
        value_t value;
        std::vector<value_t> labels(nsamples),
            values(std::size_t(nsamples) * nfeatures);

        for (const auto &file : filenames) {
            std::string line;
            std::ifstream input{file};
            if (input) {
                while (std::getline(input, line)) {
                    std::stringstream ss{line};
                    ss >> labels[rowidx];
                    while (ss >> colidx >> delim >> value)
                        values[(colidx - 1) * nsamples + rowidx] = value;
                    rowidx++;
                    if (rowidx == nsamples)
                        break;
                }
            } else {
                std::cout << "File not found\n";
            }
            if (rowidx == nsamples)
                break;
        }
        auto A = std::make_shared<matrix::dmatrix<value_t>>(
            nsamples, nfeatures, std::move(values));
        auto b = std::make_shared<const std::vector<value_t>>(std::move(labels));
        return function::loss::data<value_t>(A, b);
    }
};
} // namespace utility
#endif
