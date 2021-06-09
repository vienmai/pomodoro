cmake_minimum_required(VERSION 3.15)
project(pomodoro-test VERSION 0.1.0)

find_package(Catch2 REQUIRED)

# Loss
add_executable(logloss logloss.cpp)
target_compile_features(logloss PRIVATE cxx_std_17)
# target_link_libraries(logloss PRIVATE lib::pomodoro PRIVATE Catch2::Catch2)
target_link_libraries(logloss PRIVATE lib::pomodoro)
target_link_libraries(logloss PRIVATE Catch2::Catch2WithMain)

# Catch 
add_executable(factorial catch_example.cpp)
target_link_libraries(factorial PRIVATE Catch2::Catch2WithMain)
