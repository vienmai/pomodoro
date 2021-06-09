cmake_minimum_required(VERSION 3.15)
project(pomodoro-examples VERSION 0.1.0)

# Proxgradient
# add_executable(proxgradient proxgradient.cpp)
# target_compile_features(proxgradient PRIVATE cxx_std_17)
# target_compile_options(proxgradient PRIVATE -Wall -Wpedantic -Wno-vla-extension -O2)
# target_compile_definitions(proxgradient PUBLIC CONSTANT)
# target_link_libraries(proxgradient PRIVATE lib::pomodoro)
# # Proxgradient linesearch
# add_executable(proxgradient_linesearch proxgradient.cpp)
# target_compile_features(proxgradient_linesearch PRIVATE cxx_std_17)
# target_compile_options(proxgradient_linesearch PRIVATE -Wall -Wpedantic -Wno-vla-extension -O2)
# target_compile_definitions(proxgradient_linesearch PUBLIC LINESEARCH)
# target_link_libraries(proxgradient_linesearch PRIVATE lib::pomodoro)
# # Proxgradient Nesterov acceleration
# add_executable(acc_proxgradient proxgradient.cpp)
# target_compile_features(acc_proxgradient PRIVATE cxx_std_17)
# target_compile_options(acc_proxgradient PRIVATE -Wall -Wpedantic -Wno-vla-extension -O2)
# target_compile_definitions(acc_proxgradient PUBLIC NESTEROV_CONSTANT)
# target_link_libraries(acc_proxgradient PRIVATE lib::pomodoro)
# # Proxgradient Nesterov acceleration w/ linesearch
# add_executable(acc_proxgradient_linesearch  proxgradient.cpp)
# target_compile_features(acc_proxgradient_linesearch PRIVATE cxx_std_17)
# target_compile_options(acc_proxgradient_linesearch PRIVATE -Wall -Wpedantic -Wno-vla-extension -O2)
# target_compile_definitions(acc_proxgradient_linesearch  PUBLIC NESTEROV_LINESEARCH)
# target_link_libraries(acc_proxgradient_linesearch  PRIVATE lib::pomodoro)

# Anderson
# add_executable(anderson anderson.cpp)
# target_compile_features(anderson PRIVATE cxx_std_17)
# target_compile_options(anderson PRIVATE -Wall -Wpedantic -Wno-vla-extension -O2)
# target_link_libraries(anderson PRIVATE lib::pomodoro)

# add_executable(transform transform.cpp)
# target_compile_features(transform PRIVATE cxx_std_17)
# target_compile_options(transform PRIVATE -Wall -Wpedantic -Wno-vla-extension -O2)
# target_link_libraries(transform PRIVATE lib::pomodoro)

# add_executable(lapack_t lapack.cpp)
# target_compile_features(lapack_t PRIVATE cxx_std_17)
# target_compile_options(lapack_t PRIVATE -Wall -Wpedantic -Wno-vla-extension -O2)
# target_link_libraries(lapack_t PRIVATE lib::pomodoro)

# LBFGS
# add_executable(lbfgs lbfgs.cpp)
# target_compile_features(lbfgs PRIVATE cxx_std_17)
# target_compile_options(lbfgs PRIVATE -Wall -Wpedantic -Wno-vla-extension -O2)
# target_link_libraries(lbfgs PRIVATE lib::pomodoro)

# Pomodoro
add_executable(anderson admm.cpp)
target_compile_features(anderson PRIVATE cxx_std_17)
target_compile_options(anderson PRIVATE -Wall -Wpedantic -Wno-vla-extension -O2)
target_compile_definitions(anderson PUBLIC ANDERSON)
target_link_libraries(anderson lib::pomodoro)

# ADMM
add_executable(admm admm.cpp)
target_compile_features(admm PRIVATE cxx_std_17)
target_compile_options(admm PRIVATE -Wall -Wpedantic -Wno-vla-extension -O2)
target_compile_definitions(admm PUBLIC ADMM)
target_link_libraries(admm lib::pomodoro)

# # ANDERSON ADMM
add_executable(aa_admm admm.cpp)
target_compile_features(aa_admm PRIVATE cxx_std_17)
target_compile_options(aa_admm PRIVATE -Wall -Wpedantic -Wno-vla-extension -O2)
target_compile_definitions(aa_admm PUBLIC ANDERSON_ADMM)
target_link_libraries(aa_admm lib::pomodoro)

# APPA 
add_executable(appa admm.cpp)
target_compile_features(appa PRIVATE cxx_std_17)
target_compile_options(appa PRIVATE -Wall -Wpedantic -Wno-vla-extension -O2)
target_compile_definitions(appa PUBLIC APPA)
target_link_libraries(appa lib::pomodoro)
