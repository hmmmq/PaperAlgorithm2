add_executable(opencv_benchmark src/opencv_benchmark.cpp)
add_executable(gaussian_filter_benchmark src/gaussian_filter_benchmark.cpp)

add_dependencies(opencv_benchmark opencv_project)
add_dependencies(gaussian_filter_benchmark opencv_project)


target_include_directories(opencv_benchmark PRIVATE ${CMAKE_BINARY_DIR}/external/opencv_install/include/opencv4)
target_include_directories(gaussian_filter_benchmark PRIVATE ${CMAKE_BINARY_DIR}/external/opencv_install/include/opencv4)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(opencv_benchmark stdc++fs)
    target_link_libraries(gaussian_filter_benchmark stdc++fs)
endif()

target_link_libraries(opencv_benchmark
${OpenCV_LIB_DIR}/libopencv_core.so
${OpenCV_LIB_DIR}/libopencv_imgproc.so
${OpenCV_LIB_DIR}/libopencv_imgcodecs.so
)

target_link_libraries(gaussian_filter_benchmark
${OpenCV_LIB_DIR}/libopencv_core.so
${OpenCV_LIB_DIR}/libopencv_imgproc.so
${OpenCV_LIB_DIR}/libopencv_imgcodecs.so
)


target_link_libraries(opencv_benchmark core)
target_link_libraries(gaussian_filter_benchmark core)
