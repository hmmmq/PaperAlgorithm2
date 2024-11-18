#include <iostream>
#include <chrono>
#include "non_max_suppression.h"

void generate_test_data(double* input, double* theta, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        input[i] = rand() % 256; // 随机生成图像数据
        theta[i] = rand() % 180; // 随机生成梯度角度
    }
}

int main() {
    const int width = 1024;
    const int height = 1024;

    double *input = new double[width * height];
    double *theta = new double[width * height];
    double *output = new double[width * height];

    // 生成随机测试数据
    generate_test_data(input, theta, width, height);

    // 测试原始版本
    auto start = std::chrono::high_resolution_clock::now();
    non_max_suppression(input, output, theta, 3, width, height, 1.0);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Original non_max_suppression time: " << duration.count() << " seconds\n";

    // 测试优化版本
    start = std::chrono::high_resolution_clock::now();
    non_max_suppression_optimized(input, output, theta, 3, width, height, 1.0);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Optimized non_max_suppression time: " << duration.count() << " seconds\n";

    delete[] input;
    delete[] theta;
    delete[] output;

    return 0;
}
