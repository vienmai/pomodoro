#include <iostream>
#include <vector>

template <class value_t>
void printvec(std::vector<value_t> &x){
    for (auto &val : x)
        std::cout << " " << val << " ";
    std::cout << std::endl;
}
std::vector<double> foo(std::vector<double> x){
    std::vector<double> x_ = std::move(x);
    return x_;
}
std::vector<double> bar(std::vector<double> x) {
    return x;
}

struct vec{
    vec() = default;
    vec(size_t n) : x_(std::vector<int>(n)) { std::cout << "Constructor-size" << std::endl; }
    vec(std::vector<int> &x) : x_(x) { std::cout << "Constructor" << std::endl; }
    vec(const vec &x) : x_(x.x_) { std::cout << "Copy constructor" << std::endl; }
    vec &operator=(const vec &x) {
        std::cout << "Copy assignment" << std::endl;
        if(&x == this)
            return *this;
        x_ = std::vector<int>(x.x_.size());
        std::copy(x.x_.data(), x.x_.data() + x.x_.size(), x_.data());
        return *this;
    }
    vec(vec &&x) : x_(std::move(x.x_)) { std::cout << "Move constructor" << std::endl; };
    vec &operator=(vec &&x){
        std::cout << "Move assignment" << std::endl;
        x_ = std::move(x.x_);
        return *this;
    };

    size_t size() const {return x_.size();}
    int operator()(const int i) const {return x_[i];}
    int &operator()(const int i){return this->x_[i];}

    private:
    std::vector<int> x_;
};
inline vec operator+(const vec &a, const vec &b) {
    vec res(a.size());
    for (int i = 0; i < a.size(); i++){
        res(i) = a(i) + b(i);
    }
    return res;
}
vec add(const vec &a, const vec &b, const vec &c){
    return a + b + c;
}
vec testvec(vec x){
    std::cout << " Hi " << std::endl;
    vec z;
    z = x;
    return x;
}
vec pass_value_then_move(vec x){
    std::cout << " Hi pvtm" << std::endl;
    vec y = std::move(x);
    return y;
}
vec pass_const_ref_then_copy(const vec &x){
    std::cout << " Hi prtc" << std::endl;
    vec y = x;
    return y;
}
vec pass_const_ref_then_copy(vec &&x){
    std::cout << " Hi hihi " << std::endl;
    vec y = std::move(x);
    std::cout << " here " << std::endl;
    return y;
}

int main(int argc, char *argv[]) {
    std::vector<int> a0{1, 2, 3};
    vec a(a0);
    vec b(a);
    vec c(b);

    printf("adding\n");
    vec z = add(a, b, c); // elision, no observable copy/move constructor here
    vec o = std::move(a);

    printf("z[0] = %d\n", z(0));
    vec q = pass_const_ref_then_copy(std::move(o));

    // std::vector<int> x0{1,2,3};
    // vec x(x0);
    // vec y(x);
    // // print_vec(x); // auto copy x even if it is not used in the print_vec

    // vec z;
    // z = y;

    // vec t(std::move(z));

    // vec o;
    // o = std::move(y);

    // printf("---------------\n");
    // // vec p = testvec(std::move(o));
    // vec p = pass_value_then_move(o);
    // // vec p = pass_value_then_move(std::move(o));
    // printf("---------------\n");
    // vec q = pass_const_ref_then_copy(o);


    // std::vector<double> x0{1,2,3};
    // printvec(x0);

    // std::vector<double> x(3);
    // printvec(x);
    // x = foo(std::move(x0));

    // printvec(x);
    // printvec(x0);

    // std::cout << "--------------------" << std::endl;
    // std::vector<double> y(3);
    // y = bar(std::move(x0));
    // printvec(y);
    // printvec(x0);

    return 0;
}
