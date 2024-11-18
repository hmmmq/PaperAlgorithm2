cmake_minimum_required(VERSION 3.26)

project(FastCanny)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


option(BUILD_BENCHMARK "Build benchmark" ON)

# Enable AVX and FMA
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mfma -O3")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX /O2")
else()
  message(FATAL_ERROR "Unsupported compiler")
endif()

include(GNUInstallDirs)
include(ExternalProject)
include(CMakePrintHelpers)

# Download and build OpenCV
ExternalProject_Add(
    opencv_project
    GIT_REPOSITORY https://github.com/opencv/opencv.git
    GIT_TAG 4.10.0
    SOURCE_DIR ${CMAKE_BINARY_DIR}/external/opencv
    BINARY_DIR ${CMAKE_BINARY_DIR}/external/opencv_build
    INSTALL_DIR ${CMAKE_BINARY_DIR}/external/opencv_install
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/external/opencv_install
        -DBUILD_LIST=core,imgproc,imgcodecs
)

set(OpenCV_LIB_DIR ${CMAKE_BINARY_DIR}/external/opencv_install/${CMAKE_INSTALL_LIBDIR})

add_subdirectory(core)

if (BUILD_BENCHMARK)
  add_subdirectory(benchmark)
endif()
