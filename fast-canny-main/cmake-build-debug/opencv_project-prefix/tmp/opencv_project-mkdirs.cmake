# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/Users/meiqihuang/Public/Github Repo/PaperAlgorithm2/fast-canny-main/cmake-build-debug/external/opencv"
  "/Users/meiqihuang/Public/Github Repo/PaperAlgorithm2/fast-canny-main/cmake-build-debug/external/opencv_build"
  "/Users/meiqihuang/Public/Github Repo/PaperAlgorithm2/fast-canny-main/cmake-build-debug/external/opencv_install"
  "/Users/meiqihuang/Public/Github Repo/PaperAlgorithm2/fast-canny-main/cmake-build-debug/opencv_project-prefix/tmp"
  "/Users/meiqihuang/Public/Github Repo/PaperAlgorithm2/fast-canny-main/cmake-build-debug/opencv_project-prefix/src/opencv_project-stamp"
  "/Users/meiqihuang/Public/Github Repo/PaperAlgorithm2/fast-canny-main/cmake-build-debug/opencv_project-prefix/src"
  "/Users/meiqihuang/Public/Github Repo/PaperAlgorithm2/fast-canny-main/cmake-build-debug/opencv_project-prefix/src/opencv_project-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/meiqihuang/Public/Github Repo/PaperAlgorithm2/fast-canny-main/cmake-build-debug/opencv_project-prefix/src/opencv_project-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/meiqihuang/Public/Github Repo/PaperAlgorithm2/fast-canny-main/cmake-build-debug/opencv_project-prefix/src/opencv_project-stamp${cfgdir}") # cfgdir has leading slash
endif()
