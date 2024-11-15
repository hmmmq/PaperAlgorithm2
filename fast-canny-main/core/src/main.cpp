//
// Created by Meiqi Huang on 2024/11/15.
//
#include "gaussian_filter.h"
#include "non_maxima_suppression.h"
#include "double_threshold.h"

// 示例用法
int main() {
    int width = 1024;
    int height = 1024;
    double sigma = 1.0;
    int kernelSize = 5;

    double *inputImage = new double[width * height];  // 填充图像数据
    double *outputImage = new double[width * height];

    // 应用高斯滤波
    GaussianFilter(inputImage, outputImage, kernelSize, width, height, sigma);

    // 应用非最大抑制
    double *nmsOutput = new double[width * height];
    NonMaximaSuppression(outputImage, nmsOutput, width, height);

    // 应用双阈值
    double *thresholdOutput = new double[width * height];
    DoubleThreshold(nmsOutput, thresholdOutput, width, height, 50, 150);

    // 你可以使用OpenCV验证结果的正确性并比较性能
    delete[] inputImage;
    delete[] outputImage;
    delete[] nmsOutput;
    delete[] thresholdOutput;

    return 0;
}