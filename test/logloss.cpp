#include <catch2/catch_all.hpp>
#include "algebra.hpp"
#include "function.hpp"

TEST_CASE("Logistic loss", "[single-file]"){
    matrix::dmatrix<double> A(4, 4, {0.23745072, 1.09557628, 0.99426016, -1.19494335,
                                      1.38311306, -0.71729365, 0.93119308, -0.26876015,
                                      -0.10427326, -0.58578711, 0.92536737, -0.30543479,
                                      0.50070193, -1.79345269, 0.80630967, -0.21159942});
    std::vector<double> b{-1, 1, -1, -1};
    function::loss::data<double> data(std::move(A), b);
    function::loss::logistic_<double> loss(std::move(data)); 
    
    std::vector<double> x{0.44276126, 0.99440597, -1.02901711, 2.05714234};
    std::vector<double> g_true{-0.29389344, 2.73157336, 1.20278914, 2.85298361};
    std::vector<double> g(4);

    auto fval = loss(x.data());
    auto fval_ = loss(x.data(), g.data());

    REQUIRE(fval == Catch::Approx(8.566140387937455));
    REQUIRE(fval_ == fval);
    for (auto i = 0; i < 4; i++){
        REQUIRE(g[i] == Catch::Approx(g_true[i]));
    }
}