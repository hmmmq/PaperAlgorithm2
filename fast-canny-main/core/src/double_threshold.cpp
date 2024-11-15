//
// Created by Meiqi Huang on 2024/11/15.
//
#include "double_threshold.h"
#include <immintrin.h>
#include <iostream>

void DoubleThreshold(double *input, double *output, int width, int height,
                     double lowThreshold, double highThreshold) {

    __m256d lowTh, highTh, pixelValue, strongEdge, weakEdge, zero;

    lowTh = _mm256_set1_pd(lowThreshold);
    highTh = _mm256_set1_pd(highThreshold);
    zero = _mm256_setzero_pd();

    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j += 4) {
            // 加载当前像素值（每次加载4个像素）
            pixelValue = _mm256_loadu_pd(&input[i * width + j]);

            // 应用双阈值：强边缘、弱边缘或无边缘
            strongEdge = _mm256_cmp_pd(pixelValue, highTh, _CMP_GE_OQ);  // 强边缘
            weakEdge = _mm256_cmp_pd(pixelValue, lowTh, _CMP_GE_OQ);    // 弱边缘

            // 将强边缘设置为255（255通常用于强边缘）
            pixelValue = _mm256_blendv_pd(zero, pixelValue, strongEdge);  // 应用强边缘

            // 将弱边缘设置为1（弱边缘将在后续的滞后过程中处理）
            _mm256_storeu_pd(&output[i * width + j], pixelValue);
        }
    }
}