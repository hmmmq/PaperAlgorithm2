//
// Created by Meiqi Huang on 2024/11/15.
//
#include "non_maxima_suppression.h"

#include <cassert>
#include <immintrin.h>
#include <iostream>

void NonMaximaSuppression(double *input, double *output, int width, int height) {
    // 宽度和高度必须大于2才能应用NMS
    assert(width > 1 && height > 1);

    __m256d currentPixel, leftPixel, rightPixel, topPixel, bottomPixel;
    __m256d isMax, notMax;

    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j += 4) {
            // 每次加载4个像素（一次处理4个像素）
            currentPixel = _mm256_loadu_pd(&input[i * width + j]);

            // 加载邻居像素进行比较
            leftPixel = _mm256_loadu_pd(&input[i * width + j - 1]);
            rightPixel = _mm256_loadu_pd(&input[i * width + j + 1]);
            topPixel = _mm256_loadu_pd(&input[(i - 1) * width + j]);
            bottomPixel = _mm256_loadu_pd(&input[(i + 1) * width + j]);

            // 判断当前像素是否为邻居中的最大值
            isMax = _mm256_and_pd(
                _mm256_cmp_pd(currentPixel, leftPixel, _CMP_GT_OQ),
                _mm256_and_pd(_mm256_cmp_pd(currentPixel, rightPixel, _CMP_GT_OQ),
                               _mm256_and_pd(_mm256_cmp_pd(currentPixel, topPixel, _CMP_GT_OQ),
                                              _mm256_cmp_pd(currentPixel, bottomPixel, _CMP_GT_OQ))
                )
            );

            // 如果不是最大值，设置为0（非最大抑制）
            notMax = _mm256_xor_pd(isMax, _mm256_set1_pd(-1.0)); // 反转掩码
            currentPixel = _mm256_and_pd(currentPixel, isMax);
            _mm256_storeu_pd(&output[i * width + j], currentPixel);
        }
    }
}