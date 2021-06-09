cmake_minimum_required(VERSION 3.15)
project(pomodoro VERSION 0.1.0 LANGUAGES CXX)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)

add_library(pomodoro INTERFACE)
add_library(lib::pomodoro ALIAS pomodoro)

target_compile_features(pomodoro INTERFACE cxx_std_17)
target_compile_options(pomodoro INTERFACE -Wall -Wpedantic -Wno-vla-extension -O2)

include(GNUInstallDirs)
target_include_directories(pomodoro 
    INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(pomodoro
    INTERFACE
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    ${MPI_LIBRARIES}
)
target_link_libraries(pomodoro INTERFACE MPI::MPI_CXX)