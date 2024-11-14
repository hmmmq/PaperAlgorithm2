#include "gaussian_filter.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <iostream>

/**
 * @brief Add Padding to a matrix with a given value
 */
void PadMatrix(double *input, double *output, int width, int height,
               int padSize, int padValue) {
  int paddedWidth = width + 2 * padSize;
  int paddedHeight = height + 2 * padSize;

  std::memset(output, padValue, paddedWidth * paddedHeight * sizeof(double));

  // TODO: Should we use SIMD here?
  for (int i = 0; i < height; i++) {
    std::memcpy(output + (i + padSize) * paddedWidth + padSize,
                input + i * width, width * sizeof(double));
  }
}

/**
 * @brief Apply a Gaussian filter to an image using SIMD
 */
void GaussianFilter(double *input, double *output, int kernalSize, int width,
                    int height, double sigma) {

  // Assuming width and height are power of 2
  assert(width > 0 && (width & (width - 1)) == 0);
  assert(height > 0 && (height & (height - 1)) == 0);

  int halfSize = kernalSize / 2;
  double *kernel = new double[kernalSize * kernalSize];

  // TODO: We can try SIMD here
  GenerateGaussianKernel(kernel, kernalSize, kernalSize, sigma);

  double *paddedInput =
      new double[(width + 2 * halfSize) * (height + 2 * halfSize)];
  // Add 0 padding to the input matrix
  PadMatrix(input, paddedInput, width, height, halfSize, 0);

  int paddedWidth = width + 2 * halfSize;
  int paddedHeight = height + 2 * halfSize;

  __m256d sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10;
  __m256d pixels1, pixels2, pixels3, pixels4, pixels5, pixels6;

  __m256d kernelValue = _mm256_setzero_pd();

  // TODO: Consider cache aware optimization
  // There are 2 FMA units in ECE06 and the latency is 10
  // We need to have at least 10 FMA to max the performance
  // SO we are processing 10 * 4 elements each time
  for (int idx = 0; idx <= width * height - 40; idx += 40) {
    sum1 = _mm256_setzero_pd();
    sum2 = _mm256_setzero_pd();
    sum3 = _mm256_setzero_pd();
    sum4 = _mm256_setzero_pd();
    sum5 = _mm256_setzero_pd();
    sum6 = _mm256_setzero_pd();
    sum7 = _mm256_setzero_pd();
    sum8 = _mm256_setzero_pd();
    sum9 = _mm256_setzero_pd();
    sum10 = _mm256_setzero_pd();

    pixels1 = _mm256_setzero_pd();
    pixels2 = _mm256_setzero_pd();
    pixels3 = _mm256_setzero_pd();
    pixels4 = _mm256_setzero_pd();
    pixels5 = _mm256_setzero_pd();

    for (int k = -halfSize; k <= halfSize; k++) {
      for (int l = -halfSize; l <= halfSize; l++) {
        kernelValue = _mm256_set1_pd(
            kernel[(k + halfSize) * kernalSize + (l + halfSize)]);

        pixels1 = _mm256_loadu_pd(
            &paddedInput[((idx) / width + halfSize + k) * paddedWidth +
                         ((idx) % width + halfSize + l)]);
        pixels2 = _mm256_loadu_pd(
            &paddedInput[((idx + 4) / width + halfSize + k) * paddedWidth +
                         ((idx + 4) % width + halfSize + l)]);
        pixels3 = _mm256_loadu_pd(
            &paddedInput[((idx + 8) / width + halfSize + k) * paddedWidth +
                         ((idx + 8) % width + halfSize + l)]);
        pixels4 = _mm256_loadu_pd(
            &paddedInput[((idx + 12) / width + halfSize + k) * paddedWidth +
                         ((idx + 12) % width + halfSize + l)]);
        pixels5 = _mm256_loadu_pd(
            &paddedInput[((idx + 16) / width + halfSize + k) * paddedWidth +
                         ((idx + 16) % width + halfSize + l)]);
        sum1 = _mm256_fmadd_pd(pixels1, kernelValue, sum1);
        sum2 = _mm256_fmadd_pd(pixels2, kernelValue, sum2);
        sum3 = _mm256_fmadd_pd(pixels3, kernelValue, sum3);
        sum4 = _mm256_fmadd_pd(pixels4, kernelValue, sum4);
        sum5 = _mm256_fmadd_pd(pixels5, kernelValue, sum5);

        pixels1 = _mm256_loadu_pd(
            &paddedInput[((idx + 20) / width + halfSize + k) * paddedWidth +
                         ((idx + 20) % width + halfSize + l)]);
        pixels2 = _mm256_loadu_pd(
            &paddedInput[((idx + 24) / width + halfSize + k) * paddedWidth +
                         ((idx + 24) % width + halfSize + l)]);
        pixels3 = _mm256_loadu_pd(
            &paddedInput[((idx + 28) / width + halfSize + k) * paddedWidth +
                         ((idx + 28) % width + halfSize + l)]);
        pixels4 = _mm256_loadu_pd(
            &paddedInput[((idx + 32) / width + halfSize + k) * paddedWidth +
                         ((idx + 32) % width + halfSize + l)]);
        pixels5 = _mm256_loadu_pd(
            &paddedInput[((idx + 36) / width + halfSize + k) * paddedWidth +
                         ((idx + 36) % width + halfSize + l)]);

        sum6 = _mm256_fmadd_pd(pixels1, kernelValue, sum6);
        sum7 = _mm256_fmadd_pd(pixels2, kernelValue, sum7);
        sum8 = _mm256_fmadd_pd(pixels3, kernelValue, sum8);
        sum9 = _mm256_fmadd_pd(pixels4, kernelValue, sum9);
        sum10 = _mm256_fmadd_pd(pixels5, kernelValue, sum10);
      }
    }

    _mm256_storeu_pd(&output[((idx) / width) * width + ((idx) % width)], sum1);
    _mm256_storeu_pd(&output[((idx + 4) / width) * width + ((idx + 4) % width)],
                     sum2);
    _mm256_storeu_pd(&output[((idx + 8) / width) * width + ((idx + 8) % width)],
                     sum3);
    _mm256_storeu_pd(
        &output[((idx + 12) / width) * width + ((idx + 12) % width)], sum4);
    _mm256_storeu_pd(
        &output[((idx + 16) / width) * width + ((idx + 16) % width)], sum5);
    _mm256_storeu_pd(
        &output[((idx + 20) / width) * width + ((idx + 20) % width)], sum6);
    _mm256_storeu_pd(
        &output[((idx + 24) / width) * width + ((idx + 24) % width)], sum7);
    _mm256_storeu_pd(
        &output[((idx + 28) / width) * width + ((idx + 28) % width)], sum8);
    _mm256_storeu_pd(
        &output[((idx + 32) / width) * width + ((idx + 32) % width)], sum9);
    _mm256_storeu_pd(
        &output[((idx + 36) / width) * width + ((idx + 36) % width)], sum10);
  }

  // process the rest of the matrix
  for (int idx = (width * height / 40) * 40; idx < width * height; idx += 4) {
    sum1 = _mm256_setzero_pd();

    for (int k = -halfSize; k <= halfSize; k++) {
      for (int l = -halfSize; l <= halfSize; l++) {
        kernelValue = _mm256_set1_pd(
            kernel[(k + halfSize) * kernalSize + (l + halfSize)]);

        pixels1 = _mm256_loadu_pd(
            &paddedInput[((idx) / width + halfSize + k) * paddedWidth +
                         ((idx) % width + halfSize + l)]);

        sum1 = _mm256_fmadd_pd(pixels1, kernelValue, sum1);
      }
    }

    _mm256_storeu_pd(&output[((idx) / width) * width + ((idx) % width)], sum1);
  }

  delete[] paddedInput;
  delete[] kernel;
};

/**
 * @brief Apply a Gaussian filter to an image. This function is a slow
 * implementation of the Gaussian filter. It is used to compare the performance
 */
void GaussianFilterSlow(double *input, double *output, int kernalSize,
                        int width, int height, double sigma) {

  int halfSize = kernalSize / 2;
  double *kernel = new double[kernalSize * kernalSize];

  GenerateGaussianKernel(kernel, kernalSize, kernalSize, sigma);

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      double sum = 0.0;

      // Apply the kernel by iterating over the kernel elements
      for (int k = -halfSize; k <= halfSize; k++) {
        for (int l = -halfSize; l <= halfSize; l++) {
          int x = i + k;
          int y = j + l;
          double pixel;

          // Handle borders with zero padding
          // We can also use PadMatrix() here
          if (x < 0 || x >= height || y < 0 || y >= width) {
            pixel = 0.0;
          } else {
            pixel = input[x * width + y];
          }

          // Calculated the weighted sum
          sum += pixel * kernel[(k + halfSize) * kernalSize + (l + halfSize)];
        }
      }

      output[i * width + j] = sum;
    }
  }

  delete[] kernel;
};

/**
 * @brief Generate a Gaussian kernel
 */
void GenerateGaussianKernel(double *kernel, int width, int height,
                            double sigma) {

  int halfWidth = width / 2;
  double sum = 0.0;

  for (int x = -halfWidth; x <= halfWidth; x++) {
    for (int y = -halfWidth; y <= halfWidth; y++) {
      // https://en.wikipedia.org/wiki/Gaussian_filter
      double value = std::exp(-(x * x + y * y) / (2 * sigma * sigma)) /
                     (2 * M_PI * sigma * sigma);
      kernel[(x + halfWidth) * width + (y + halfWidth)] = value;
      sum += value;
    }
  }

  for (int i = 0; i < width * height; i++) {
    kernel[i] /= sum;
  }
};
