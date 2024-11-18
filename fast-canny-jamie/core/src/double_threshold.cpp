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
            pixelValue = _mm256_loadu_pd(&input[i * width + j]);

            strongEdge = _mm256_cmp_pd(pixelValue, highTh, _CMP_GE_OQ);
            weakEdge = _mm256_cmp_pd(pixelValue, lowTh, _CMP_GE_OQ);

            pixelValue = _mm256_blendv_pd(zero, pixelValue, strongEdge);

            _mm256_storeu_pd(&output[i * width + j], pixelValue);
        }
    }
}

void DoubleThresholdSlow(double *input, double *output, int height, int width, double lowThreshold, double highThreshold){
    for(int i=1; i<height-1; i++){
        for(int j=1; j<width-1; j++){
            if(input[i][j] >= highThreshold){
                output[i][j] = highThreshold;
            }
            else if(lowThreshold <= input[i][j] && input[i][j] <= highThreshold){
                output[i][j] = lowThreshold;
            }
            else{
                output[i][j] = 0;
            }
        }
    }
    return;
}