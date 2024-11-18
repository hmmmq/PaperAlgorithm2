#include <iostream>
#include <vector>
#include "non_max_suppression.h"

// 辅助函数，用于打印图像数组
void printArray(const std::vector<double>& array, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << array[i * width + j] << "\t";
        }
        std::cout << std::endl;
    }
}

int main() {
    // 图像宽度和高度
    int width = 5;
    int height = 5;

    // 初始化测试输入数据
    std::vector<double> input = {
        10, 20, 30, 40, 50,
        15, 25, 35, 45, 55,
        20, 30, 50, 40, 30,
        25, 35, 45, 55, 65,
        30, 40, 50, 60, 70
    };

    // 梯度方向数组（模拟数据，单位为弧度）
    std::vector<double> theta = {
        0.0, 0.2, 0.4, 0.6, 0.8,
        0.2, 0.4, 0.6, 0.8, 1.0,
        0.4, 0.6, 1.57, 0.8, 0.6,
        0.6, 0.8, 1.0, 0.4, 0.2,
        0.8, 1.0, 1.2, 1.4, 1.6
    };

    // 输出数组，初始化为全零
    std::vector<double> output(width * height, 0);

    // 内核大小（非极大值抑制中未使用）
    int kernelSize = 3;

    // sigma（非极大值抑制中未使用）
    double sigma = 1.0;

    // 调用 non_max_suppression 函数
    non_max_suppression(input.data(), output.data(), theta.data(), kernelSize, width, height, sigma);

    // 打印结果
    std::cout << "Input Image:" << std::endl;
    printArray(input, width, height);

    std::cout << "\nTheta (Gradient Directions):" << std::endl;
    printArray(theta, width, height);

    std::cout << "\nOutput Image (After Non-Max Suppression):" << std::endl;
    printArray(output, width, height);

    return 0;
}
