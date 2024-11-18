#include <iostream>
#include <chrono> // 用于计时
#include <random> // 用于生成随机数据
#include "double_threshold.h"

void generate_random_data(double *data, int size, double min_val, double max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min_val, max_val);

    for (int i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

int main() {
    int width = 1920;  // 图像宽度
    int height = 1080; // 图像高度
    int size = width * height;

    // 分配内存
    double *input = new double[size];
    double *output_original = new double[size];
    double *output_optimized = new double[size];

    // 生成随机输入数据
    generate_random_data(input, size, 0, 255);

    // 定义阈值
    double low_thres = 50;
    double high_thres = 100;

    // 计时原始版本
    auto start_original = std::chrono::high_resolution_clock::now();
    double_threshold(input, output_original, width, height, low_thres, high_thres);
    auto end_original = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_original = end_original - start_original;
    std::cout << "Original version took: " << duration_original.count() << " seconds.\n";

    // 计时优化版本
    auto start_optimized = std::chrono::high_resolution_clock::now();
    double_threshold_optimized(input, output_optimized, width, height, low_thres, high_thres);
    auto end_optimized = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_optimized = end_optimized - start_optimized;
    std::cout << "Optimized version took: " << duration_optimized.count() << " seconds.\n";

    // 检查结果是否一致
    bool is_correct = true;
    for (int i = 0; i < size; i++) {
        if (output_original[i] != output_optimized[i]) {
            is_correct = false;
            std::cout << "Mismatch at index " << i << ": original=" << output_original[i]
                      << ", optimized=" << output_optimized[i] << "\n";
            break;
        }
    }

    if (is_correct) {
        std::cout << "Optimized version produces the same results as the original.\n";
    } else {
        std::cout << "Results differ between the two versions.\n";
    }

    // 释放内存
    delete[] input;
    delete[] output_original;
    delete[] output_optimized;

    return 0;
}
