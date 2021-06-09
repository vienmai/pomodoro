#ifndef UTILITY_LOGGER_HPP_
#define UTILITY_LOGGER_HPP_

#include <chrono>
#include <cstddef>
#include <iterator>
#include <memory>
#include <mutex>
#include <ostream>
#include <type_traits>
#include <vector>

namespace utility {
namespace logger {
namespace detail {
template<class T, class container>
struct iterator_t {
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = value_type*;
    using reference = value_type&;
    using iterator_category = std::random_access_iterator_tag;

    iterator_t() = default;
    iterator_t(const container *lptr, const std::ptrdiff_t n) : lptr{lptr}, n{n}{
        iterate = lptr->setiterate(n);
    }

    reference operator*() { return iterate; }
    value_type const &operator*() const { return iterate; }
    pointer operator->() { return &iterate; }
    value_type const *operator->() const { return &iterate; }
    bool operator==(const iterator_t &rhs) const {
        return n == rhs.n && lptr == rhs.lptr;
    }
    bool operator!=(const iterator_t &rhs) const { return !operator==(rhs); }
    iterator_t &operator++() {
        n++;
        iterate = lptr->setiterate(n);
        return *this;
    }
    iterator_t operator++(int) {
        iterator_t ip{*this};
        operator++();
        return ip;
    }
    iterator_t &operator--() {
        n--;
        iterate = lptr->setiterate(n);
        return *this;
    }
    iterator_t operator--(int) {
        iterator_t ip{*this};
        operator--();
        return ip;
    }
    iterator_t &operator+=(const difference_type n) {
        this->n += n;
        iterate = lptr->setiterate(this->n);
        return *this;
    }
    friend iterator_t operator+(const iterator_t it, const difference_type n) {
        iterator_t temp{it};
        return temp += n;
    }
    friend iterator_t operator+(const difference_type n, const iterator_t it) {
        return it + n;
    }
    iterator_t &operator-=(const difference_type n) {
        this->n += -n;
        iterate = lptr->setiterate(this->n);
        return *this;
    }
    friend iterator_t operator-(const iterator_t it, const difference_type n) {
        iterator_t temp{it};
        return temp -= n;
    }
    difference_type operator-(const iterator_t rhs) const { return n - rhs.n; }
    value_type operator[](const difference_type n) const {
        return *(*this + n);
    }
    bool operator<(const iterator_t rhs) const { return rhs.n - n > 0; }
    bool operator>(const iterator_t rhs) const { return rhs < *this; }
    bool operator>=(const iterator_t rhs) const { return !operator<(rhs); }
    bool operator<=(const iterator_t rhs) const { return !operator>(rhs); }

private:
    const container *lptr{nullptr};
    std::ptrdiff_t n{0};
    value_type iterate;
};

template <class value_t, bool log_x_v, bool log_feas>
class logger {
    struct iterate_t {
        iterate_t() = default;
        iterate_t(const int *k, const value_t *t, const value_t *f)
            : k{k}, t{t}, f{f} {}
        iterate_t(const int *k, const value_t *t, const value_t *f,
                  const value_t *feas)
            : k{k}, t{t}, f{f}, feas{feas} {}
        iterate_t(const int *k, const value_t *t, const value_t *f,
                  const std::vector<value_t> *x)
            : k{k}, t{t}, f{f}, x{x} {}
        iterate_t(const int *k, const value_t *t, const value_t *f,
                  const value_t *feas, const std::vector<value_t> *x)
            : k{k}, t{t}, f{f}, feas{feas}, x{x} {}

        const int &getk() const { return *k; }
        const value_t &gett() const { return *t; }
        const value_t &getf() const { return *f; }
        const value_t &getfeas() const {
            // static_assert(log_feas, "getfeas is only defined for constrained optimization");
            return *feas; 
        }
        const std::vector<value_t> &getx() const {
            static_assert(log_x_v, "getx is only defined for full and decision loggers");
            return *x;
        }
        iterate_t &delimiter(const char delim) {
            this->delim = delim;
            return *this;
        }
        friend std::ostream &operator<<(std::ostream &os, const iterate_t &iter) {
            os << *iter.k << iter.delim << *iter.t << iter.delim << *iter.f;
            return os;
        }

        private:
            char delim{','};
            const int *k{nullptr};
            const value_t *t{nullptr};
            const value_t *f{nullptr};
            const value_t *feas{nullptr};
            const std::vector<value_t> *x{nullptr};
    };
    template <class Container, class InputIt>
    void log(Container &c, InputIt begin, InputIt end, std::true_type) {
        c.emplace_back(begin, end);
    }
    template <class Container, class InputIt>
    void log(Container &, InputIt, InputIt, std::false_type) {}

    std::unique_ptr<std::mutex> sync{new std::mutex{}};
    std::vector<int> iterations;
    std::vector<value_t> times;
    std::vector<value_t> fvalues;
    std::vector<value_t> feasibility; // should consider std::reserve(max_iters)
    std::vector<std::vector<value_t>> xvalues; // and provide max_iters in the logger constructor
    std::chrono::time_point<std::chrono::high_resolution_clock> tstart{
        std::chrono::high_resolution_clock::now()}, tend;

public:
    using value_type = iterate_t;
    using iterator_t = detail::iterator_t<iterate_t, logger>;
    iterator_t begin() const { return iterator_t(this, 0); }
    iterator_t end() const { return iterator_t(this, iterations.size()); }

    template <class InputIt>
    void operator()(const int k, const value_t fval,
                    InputIt xbegin, InputIt xend){
        std::lock_guard<std::mutex> lock(*sync);
        tend = std::chrono::high_resolution_clock::now();
        const auto telapsed =
            std::chrono::duration<value_t, std::chrono::seconds::period>(
                tend - tstart);
        iterations.push_back(k);
        times.push_back(telapsed.count());
        fvalues.push_back(fval);
        log(xvalues, xbegin, xend, std::integral_constant<bool, log_x_v>{});
    }
    template <class InputIt>
    void operator()(const int k, const value_t fval, const value_t feas,
                    InputIt xbegin, InputIt xend){
        this->operator()(k, fval, xbegin, xend);
        feasibility.push_back(feas);
    }
    iterate_t setiterate(const std::ptrdiff_t n) const noexcept {
        if (std::integral_constant<bool, log_x_v>{})
            return std::integral_constant<bool, log_feas>{} ? 
                   iterate_t{&iterations[0] + n, &times[0] + n, &fvalues[0] + n,                                                    
                             &feasibility[0] + n, &xvalues[0] + n} :
                   iterate_t{&iterations[0] + n, &times[0] + n, &fvalues[0] + n, &xvalues[0] + n};
        return std::integral_constant<bool, log_feas>{} ? 
               iterate_t{&iterations[0] + n, &times[0] + n, &fvalues[0] + n, &feasibility[0] + n} :
               iterate_t{&iterations[0] + n, &times[0] + n, &fvalues[0] + n};
    }
    void csv(const std::string &filename) {
        std::cout << filename << '\n';
        std::ofstream file(filename);
        if (file){
            if (!std::integral_constant<bool, log_feas>{}){
                file << "k,t,f\n";
                for (const auto &log : *this)
                    file << std::fixed << log.getk() << ','
                         << log.gett() << ',' << log.getf() << '\n';
            }
            else{
                file << "k,t,f,feas\n";
                for (const auto &log : *this)
                    file << std::fixed << log.getk() << ',' << log.gett() << ','
                         << log.getf() << ',' << log.getfeas() << '\n';
            }
        } else {
            std::cout << "File not found\n";
        }
    }
};
} // namespace detail

template <class value_t = double>
using value = detail::logger<value_t, false, false>;
template <class value_t = double>
using decision = detail::logger<value_t, true, false>;
template <class value_t = double>
using valfeas = detail::logger<value_t, false, true>;
template <class value_t = double>
using full = detail::logger<value_t, true, true>;
} // namespace logger
} // namespace utility

#endif
