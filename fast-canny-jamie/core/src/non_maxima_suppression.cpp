#include "non_maxima_suppression.h"

#include <cassert>
#include <immintrin.h>
#include <iostream>

void NonMaximaSuppression(double *input, double *output, double *theta, int width, int height) {

    __m256 currentPixel, neighbor1, neighbor2;
    __m256 isMax, zero = _mm256_set1_ps(0.0f);

    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j += 8) {
            currentPixel = _mm256_loadu_ps(&input[i * width + j]);
            __m256 angle = _mm256_loadu_ps(&theta[i * width + j]);

            // Normalize angle to [0, 180)
            angle = _mm256_mul_ps(angle, _mm256_set1_ps(180.0f / M_PI));
            angle = _mm256_andnot_ps(_mm256_cmp_ps(angle, _mm256_set1_ps(0.0f), _CMP_LT_OQ), angle);

            __m256 mask1 = _mm256_or_ps(
                _mm256_cmp_ps(angle, _mm256_set1_ps(22.5f), _CMP_LT_OQ),
                _mm256_cmp_ps(angle, _mm256_set1_ps(157.5f), _CMP_GE_OQ)
            );
            neighbor1 = _mm256_loadu_ps(&input[i * width + j + 1]);
            neighbor2 = _mm256_loadu_ps(&input[i * width + j - 1]);

            // Neighbor selection for 45°
            __m256 mask2 = _mm256_and_ps(
                _mm256_cmp_ps(angle, _mm256_set1_ps(22.5f), _CMP_GE_OQ),
                _mm256_cmp_ps(angle, _mm256_set1_ps(67.5f), _CMP_LT_OQ)
            );
            neighbor1 = _mm256_blendv_ps(neighbor1, _mm256_loadu_ps(&input[(i + 1) * width + j - 1]), mask2);
            neighbor2 = _mm256_blendv_ps(neighbor2, _mm256_loadu_ps(&input[(i - 1) * width + j + 1]), mask2);

            // Neighbor selection for 90°
            __m256 mask3 = _mm256_and_ps(
                _mm256_cmp_ps(angle, _mm256_set1_ps(67.5f), _CMP_GE_OQ),
                _mm256_cmp_ps(angle, _mm256_set1_ps(112.5f), _CMP_LT_OQ)
            );
            neighbor1 = _mm256_blendv_ps(neighbor1, _mm256_loadu_ps(&input[(i + 1) * width + j]), mask3);
            neighbor2 = _mm256_blendv_ps(neighbor2, _mm256_loadu_ps(&input[(i - 1) * width + j]), mask3);

            // Neighbor selection for 135°
            __m256 mask4 = _mm256_and_ps(
                _mm256_cmp_ps(angle, _mm256_set1_ps(112.5f), _CMP_GE_OQ),
                _mm256_cmp_ps(angle, _mm256_set1_ps(157.5f), _CMP_LT_OQ)
            );
            neighbor1 = _mm256_blendv_ps(neighbor1, _mm256_loadu_ps(&input[(i - 1) * width + j - 1]), mask4);
            neighbor2 = _mm256_blendv_ps(neighbor2, _mm256_loadu_ps(&input[(i + 1) * width + j + 1]), mask4);

            // Check if the current pixel is a local maximum
            isMax = _mm256_and_ps(
                _mm256_cmp_ps(currentPixel, neighbor1, _CMP_GE_OQ),
                _mm256_cmp_ps(currentPixel, neighbor2, _CMP_GE_OQ)
            );

            // Suppress non-maxima
            __m256 result = _mm256_blendv_ps(zero, currentPixel, isMax);

            // Store the result
            _mm256_storeu_ps(&output[i * width + j], result);
        }
    }
}

void NonMaximaSuppressionSlow(double *input, double *output, double *theta, int height, int width) {
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            int idx = i * width + j; // Calculate the index for the flattened array

            // Normalize theta to [0, 180)
            theta[idx] = theta[idx] * 180 / M_PI;
            if (theta[idx] < 0) {
                theta[idx] += 180;
            }

            double q = 255.0;
            double r = 255.0;

            // Determine neighbors based on theta
            if ((0 <= theta[idx] && theta[idx] < 22.5) || (157.5 <= theta[idx] && theta[idx] <= 180)) {
                q = input[idx + 1];     // Right neighbor
                r = input[idx - 1];     // Left neighbor
            } else if (22.5 <= theta[idx] && theta[idx] < 67.5) {
                q = input[(i + 1) * width + j - 1]; // Bottom-left neighbor
                r = input[(i - 1) * width + j + 1]; // Top-right neighbor
            } else if (67.5 <= theta[idx] && theta[idx] < 112.5) {
                q = input[(i + 1) * width + j];     // Bottom neighbor
                r = input[(i - 1) * width + j];     // Top neighbor
            } else if (112.5 <= theta[idx] && theta[idx] < 157.5) {
                q = input[(i - 1) * width + j - 1]; // Top-left neighbor
                r = input[(i + 1) * width + j + 1]; // Bottom-right neighbor
            }

            // Check if the current pixel is a local maximum
            if (input[idx] >= q && input[idx] >= r) {
                output[idx] = input[idx];
            } else {
                output[idx] = 0.0;
            }
        }
    }
}