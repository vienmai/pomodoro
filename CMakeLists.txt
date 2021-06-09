cmake_minimum_required(VERSION 3.15)

project(pomodoro-super LANGUAGES CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CMakePrintHelpers)
cmake_print_variables(CMAKE_CXX_COMPILER CMAKE_MAJOR_VERSION)

add_subdirectory(src)
add_subdirectory(examples)

enable_testing()
add_subdirectory(test)