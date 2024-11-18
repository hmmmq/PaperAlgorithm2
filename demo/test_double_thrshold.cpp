//
// Created by Meiqi Huang on 24-11-18.
//
#include "double_threshold.h"

// 源文件中实现 double_threshold 函数...
#include <iostream>

int main() {
    int width = 5, height = 5;
    double input[] = {10, 55, 120, 80, 30,
                      60, 95, 140, 50, 20,
                      70, 110, 200, 40, 15,
                      20, 30, 90, 60, 10,
                      5,  25, 70, 35, 15};
    double output[25] = {0}; // 初始化输出数组
    double low_thres = 50, high_thres = 100;

    double_threshold(input, output, 0, width, height, 0.0, low_thres, high_thres);

    // 打印输出结果
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << output[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}