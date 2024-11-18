cmake_minimum_required(VERSION 3.29)
project(demo)

set(CMAKE_CXX_STANDARD 20)

#add_executable(demo main.cpp
#        double_threshold.cpp
#        non_max_suppression.cpp
#        double_threshold.h
#        non_max_suppression.h
#        benchmark_double_thrshold.cpp
#        benchmark_non_maxima_suppression.cpp)
# Add executable for main.cpp
add_executable(demo_main main.cpp )

# Add executable for benchmark_double_thrshold.cpp
add_executable(benchmark_threshold test_double_thrshold.cpp double_threshold.cpp)

# Add executable for benchmark_non_maxima_suppression.cpp
add_executable(benchmark_nms test_non_max_suppression.cpp non_max_suppression.cpp)

add_executable(benchmark_threshold2 benchmark_double_thrshold.cpp double_threshold.cpp)

add_executable(benchmark_nms2 benchmark_non_max_suppression.cpp non_max_suppression.cpp)


add_executable(test testavx2intrics.h testavx2intrics.cpp)
