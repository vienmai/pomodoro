#include <iostream>
#include <map>
#include <array>

int main(int argc, char *argv[])
{
    // std::map<std::string, int> user_age{{"Dut", 30}, {"Teo", 29}};
    // std::map mycopy(user_age);
    // if (auto [iter, was_added] = mycopy.insert_or_assign("Teo", 30); !was_added)
    //     std::cout << iter->first << "re-assigned...\n";
    // for (const auto &[key, val] : mycopy){
    //     std::cout << key << ',' << val << '\n';
    // }
    std::map<std::string, std::array<size_t, 2> > dataset{
        {"heart", {270, 13}},
        {"a8a", {22696, 123}}};

    std::pair<std::string, std::array<size_t, 2> > dataset_;
    dataset_ = {"heart__", {270, 13}};

    // for (const auto &[key, val] : dataset_) {
    //     std::cout << key << ',' << val[0] << ',' << val[1] << '\n';
    // }
    // std::string name = "abc";
    // dataset_.emplace(std::make_pair(name, std::array<size_t, 2>{{100, 10}}));
    // for (const auto &[key, val] : dataset_){
    //     std::cout << key << ',' << val[0] << ',' << val[1] << '\n';
    // }
    const auto &[filename, dims] = dataset_;
    std::cout << filename << ',' << dims[0] << ',' << dims[1] << '\n';
}