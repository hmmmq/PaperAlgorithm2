add_library(core STATIC src/gaussian_filter.cpp)

add_dependencies(core opencv_project)

target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(core PRIVATE ${CMAKE_BINARY_DIR}/external/opencv_install/include/opencv4)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(core stdc++fs)
endif()

target_link_libraries(core
${OpenCV_LIB_DIR}/libopencv_core.so
${OpenCV_LIB_DIR}/libopencv_imgproc.so
${OpenCV_LIB_DIR}/libopencv_imgcodecs.so
)
