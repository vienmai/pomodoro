#ifndef STEPSIZE_CONSTANT_HPP_
#define STEPSIZE_CONSTANT_HPP_

namespace stepsize {
template <class value_t> struct constant : public astepsize<value_t> {
    constant() = default;

    template <class Loss, class Prox, class InputIt1, class InputIt2>
    value_t get_stepsize(Loss &&loss, Prox &&prox, const int k, const value_t fx,
                         InputIt1 xbegin, InputIt1 xend, InputIt2 gcurr){
        return stepsize;
    }

protected:
    void parameters(const value_t stepsize) { this->stepsize = stepsize; }
private:
    value_t stepsize;
};
} // namespace step

#endif
