#ifndef HELPER_HPP_
#define HELPER_HPP_

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include "../matrix/dmatrix.hpp"
#include "../utility/random.hpp"

namespace algebra{
using namespace algebra::matrix;
using namespace utility;
template<class value_t>
value_t vdot(const value_t *x, const std::vector<value_t> &y) {
	return blas<value_t>::dot(y.size(), x, 1, y.data(), 1);
}
template<class value_t>
value_t vdot(const std::vector<value_t> &x, const std::vector<value_t> &y) {
        return blas<value_t>::dot(x.size(), x.data(), 1, y.data(), 1);
}
template<class value_t>
value_t ltwo(const std::vector<value_t> &x) {
	return blas<value_t>::nrm2(x.size(), x.data(), 1);
}
template<class value_t>
void sub(const value_t *x, const std::vector<value_t> &y, std::vector<value_t> &out) {
	auto yb = std::begin(y);
	auto ye = std::end(y);
	auto outb = std::begin(out);
	while (yb != ye)
		*outb++ = *x++ - *yb++;
}
template <class value_t>
void sub(std::vector<value_t> &x, std::vector<value_t> &y, std::vector<value_t> &out){
	std::transform(std::begin(x), std::end(x), std::begin(y), std::begin(out),
				   [](const value_t xval, const value_t yval) { return xval - yval; });
}
template <class value_t>
std::vector<value_t> sub(std::vector<value_t> &x, std::vector<value_t> &y){
	std::vector<value_t> res(x.size());
	std::transform(std::begin(x), std::end(x), std::begin(y), std::begin(res),
				   [](const value_t xval, const value_t yval) { return xval - yval; });
	return res;
}
template <class value_t>
void add(std::vector<value_t> &x, std::vector<value_t> &y, std::vector<value_t> &out){
	std::transform(std::begin(x), std::end(x), std::begin(y), std::begin(out),
				   [](const value_t xval, const value_t yval) { return xval + yval; });
}
template<class value_t>
void add(const value_t *x, const std::vector<value_t> &y, std::vector<value_t> &out) {
	auto yb = std::begin(y);
	auto ye = std::end(y);
	auto outb = std::begin(out);
	while (yb != ye)
		*outb++ = *x++ + *yb++;
}
template <class value_t>
void add(const value_t *x, const int n, const value_t *y, value_t *out) {
	auto xe = x + n;
	while (x != xe)
		*out++ = *x++ + *y++;
}
}//namespace algebra
#endif
