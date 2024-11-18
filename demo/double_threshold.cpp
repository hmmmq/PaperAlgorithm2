#include <cmath>
#include <algorithm>

void double_threshold(double *input, double *output, int kernalSize,
                      int width, int height, double sigma,
                      double low_thres = 50, double high_thres = 100) {
    // 遍历图像数据（假设 input 是一维数组，按行主序存储）
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int idx = i * width + j; // 计算一维数组中的索引
            if (input[idx] >= high_thres) {
                output[idx] = high_thres; // 强边缘
            } else if (input[idx] >= low_thres && input[idx] <= high_thres) {
                output[idx] = low_thres; // 弱边缘
            } else {
                output[idx] = 0; // 非边缘
            }
        }
    }
}

#include <immintrin.h>  // For AVX2 intrinsics

void double_threshold_optimized(double *input, double *output, int width, int height,
                                double low_thres = 50, double high_thres = 100) {
    // 图像的总大小
    int size = width * height;

    // 使用 SIMD 操作处理数据
    int i = 0;

    // 处理 4 个 double 元素一组（每组 32 字节）
    for (; i <= size - 4; i += 4) {
        // 使用 _mm256_loadu_pd 从内存中加载 4 个 double 元素到 AVX2 寄存器
        __m256d input_vals = _mm256_loadu_pd(&input[i]);

        // 创建一个常量寄存器来存储阈值 low_thres 和 high_thres
        __m256d low_vals = _mm256_set1_pd(low_thres);
        __m256d high_vals = _mm256_set1_pd(high_thres);

        // 将 input_vals 与 high_thres 和 low_thres 进行比较
        __m256d high_mask = _mm256_cmp_pd(input_vals, high_vals, _CMP_GE_OS);  // input[i] >= high_thres
        __m256d low_mask = _mm256_andnot_pd(high_mask, _mm256_cmp_pd(input_vals, low_vals, _CMP_GE_OS));  // low_thres <= input[i] < high_thres

        // 使用 _mm256_blendv_pd 根据比较结果选择高阈值或低阈值或 0
        __m256d result = _mm256_blendv_pd(
            _mm256_setzero_pd(),  // 如果都不满足则为 0
            high_vals,             // 满足高阈值条件时设置为 high_thres
            high_mask
        );

        result = _mm256_blendv_pd(result, low_vals, low_mask);  // 满足低阈值条件时设置为 low_thres

        // 使用 _mm256_storeu_pd 将计算结果存回 output 数组
        _mm256_storeu_pd(&output[i], result);
    }

    // 处理剩下的少于 4 个像素（如果 size 不是 4 的倍数）
    for (; i < size; i++) {
        if (input[i] >= high_thres) {
            output[i] = high_thres;
        } else if (input[i] >= low_thres) {
            output[i] = low_thres;
        } else {
            output[i] = 0;
        }
    }
}
