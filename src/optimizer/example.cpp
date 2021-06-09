#include <iostream>
#include <vector>
 
int main()
{
    // Create a vector containing integers
    std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int> g = {0, 500};
    
    auto n = 2;
    auto start = v.data() + 1*n;
    std::cout << start << '\n';

    std::transform(start, start + n, g.data(), start,
                   [](const int vval, const int gxval) { return gxval; });
    // Print out the vector
    std::cout << "v = { ";
    for (int n : v) {
        std::cout << n << ", ";
    }
    std::cout << "}; \n";
}