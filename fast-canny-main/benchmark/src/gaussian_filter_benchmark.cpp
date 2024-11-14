#include "gaussian_filter.h"
#include "opencv2/core/base.hpp"
#include "opencv2/core/mat.hpp"
#include <exception>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <stdexcept>

#define MAX_FREQ 3.4
#define BASE_FREQ 2.4
#define NUM_IMAGES_PER_SIZE 10
#define GAUSSIAN_KERNEL_SIZE 3
#define GAUSSIAN_KERNEL_SIGMA 0.5
#define CANNY_GRADIENT_LOWER_THRESHOLD 100
#define CANNY_GRADIENT_UPPER_THRESHOLD 200

static inline unsigned long long rdtsc(void) {
  unsigned hi, lo;
  asm volatile("rdtsc" : "=a"(lo), "=d"(hi));
  return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

void TestMatrixPadding() {
  double input[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  double *output = new double[25]();
  PadMatrix(input, output, 3, 3, 1, 0);

  double expected[25] = {0, 0, 0, 0, 0, 0, 0, 1, 2, 0,
                         0, 3, 4, 5, 0, 0, 6, 7, 8, 0};

  for (int i = 0; i < 25; i++) {
    if (output[i] != expected[i]) {
      std::cout << "output[" << i << "] = " << output[i]
                << " expected: " << expected[i] << "\n";
      throw std::runtime_error("TestMatrixPadding failed");
    }
  }

  delete[] output;
};

void BenchmarkGaussianFilterSlow(int width, int height) {
  unsigned long long st;
  unsigned long long et;
  unsigned long long total = 0;
  unsigned long long referenceTotal = 0;
  int repeat = 1000;
  int matrixSize = width * height;

  int lower_bound = 0;
  int upper_bound = 256;

  std::uniform_int_distribution<int> unif(lower_bound, upper_bound);

  std::default_random_engine re;

  double *input = new double[matrixSize]();
  double *output = new double[matrixSize]();

  for (int i = 0; i < matrixSize; i++) {
    input[i] = unif(re);
    output[i] = 0.0;
  }

  cv::Mat src(height, width, CV_64F, input);
  cv::Mat expected;

  GaussianFilterSlow(input, output, GAUSSIAN_KERNEL_SIZE, width, height,
                     GAUSSIAN_KERNEL_SIGMA);
  cv::GaussianBlur(src, expected,
                   cv::Size(GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE),
                   GAUSSIAN_KERNEL_SIGMA, cv::BORDER_CONSTANT, 0);

  for (int i = 0; i < matrixSize; i++) {
    if (std::abs(output[i] - expected.at<double>(i)) > 1e-6) {
      std::cout << "output[" << i << "] = " << output[i]
                << " expected: " << expected.at<double>(i) << "\n";
      throw std::runtime_error("BenchmarkGaussianFilterSlow  failed: incorrect "
                               "output from GaussianFilterSlow");
    }
  }

  for (int i = 0; i != repeat; ++i) {
    st = rdtsc();
    GaussianFilterSlow(input, output, GAUSSIAN_KERNEL_SIZE, width, height,
                       GAUSSIAN_KERNEL_SIGMA);
    et = rdtsc();

    total += (et - st);
  }

  for (int i = 0; i != repeat; ++i) {
    st = rdtsc();
    cv::GaussianBlur(src, expected,
                     cv::Size(GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE),
                     GAUSSIAN_KERNEL_SIGMA, cv::BORDER_CONSTANT, 0);
    et = rdtsc();

    referenceTotal += (et - st);
  }

  unsigned long long createFilterFLOPSPS = (4 + 6 + 1 + 1 + 1) * 9;
  unsigned long long kernalFLOPSPS =
      2 * 9 * width * height + createFilterFLOPSPS;

  std::cout << "Benchmarking matrix size: " << width << "x" << height << "\n";
  std::cout << "RDTSC Cycles Taken for GaussianFilterSlow: " << total << "\n";
  std::cout << "RDTSC Cycles Taken for cv::GaussianBlur: " << referenceTotal
            << "\n";
  std::cout << "FLOPS Per Cycle for GaussianFilterSlow: "
            << repeat * kernalFLOPSPS / (total * MAX_FREQ / BASE_FREQ) << "\n";
  std::cout << "FLOPS Per Cycle for cv::GaussianBlur: "
            << repeat * kernalFLOPSPS / (referenceTotal * MAX_FREQ / BASE_FREQ)
            << "\n";

  delete[] input;
  delete[] output;
}

void BenchmarkGaussianFilter(int width, int height) {
  unsigned long long st;
  unsigned long long et;
  unsigned long long total = 0;
  unsigned long long referenceTotal = 0;
  int repeat = 1000;
  int matrixSize = width * height;

  int lower_bound = 0;
  int upper_bound = 256;

  std::uniform_int_distribution<int> unif(lower_bound, upper_bound);
  std::default_random_engine re;

  double *input = new double[matrixSize]();
  double *output = new double[matrixSize]();

  // Generate a random input matrix
  for (int i = 0; i < matrixSize; i++) {
    input[i] = unif(re);
    output[i] = 0.0;
  }

  cv::Mat src(height, width, CV_64F, input);
  cv::Mat expected;

  GaussianFilter(input, output, GAUSSIAN_KERNEL_SIZE, width, height,
                 GAUSSIAN_KERNEL_SIGMA);
  cv::GaussianBlur(src, expected,
                   cv::Size(GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE),
                   GAUSSIAN_KERNEL_SIGMA, cv::BORDER_CONSTANT, 0);

  // Check if the output is correct
  for (int i = 0; i < matrixSize; i++) {
    if (std::abs(output[i] - expected.at<double>(i)) > 1e-6) {
      std::cout << "output[" << i << "] = " << output[i]
                << " expected: " << expected.at<double>(i) << "\n";
      throw std::runtime_error("BenchmarkGaussianFilter failed: incorrect "
                               "output from GaussianFilter");
    }
  }

  for (int i = 0; i != repeat; ++i) {
    st = rdtsc();
    GaussianFilter(input, output, GAUSSIAN_KERNEL_SIZE, width, height,
                   GAUSSIAN_KERNEL_SIGMA);
    et = rdtsc();

    total += (et - st);
  }

  for (int i = 0; i != repeat; ++i) {
    st = rdtsc();
    cv::GaussianBlur(src, expected,
                     cv::Size(GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE),
                     GAUSSIAN_KERNEL_SIGMA, cv::BORDER_CONSTANT, 0);
    et = rdtsc();

    referenceTotal += (et - st);
  }

  unsigned long long createFilterFLOPSPS = (4 + 6 + 1 + 1 + 1) * 9;
  unsigned long long kernalFLOPSPS =
      2 * 9 * width * height + createFilterFLOPSPS;

  std::cout << "Benchmarking matrix size: " << width << "x" << height << "\n";
  std::cout << "RDTSC Cycles Taken for GaussianFilter: " << total << "\n";
  std::cout << "RDTSC Cycles Taken for cv::GaussianBlur: " << referenceTotal
            << "\n";
  std::cout << "FLOPS Per Cycle for GaussianFilter: "
            << repeat * kernalFLOPSPS / (total * MAX_FREQ / BASE_FREQ) << "\n";
  std::cout << "FLOPS Per Cycle for cv::GaussianBlur: "
            << repeat * kernalFLOPSPS / (referenceTotal * MAX_FREQ / BASE_FREQ)
            << "\n";
}

void TestGaussianFilterSlowCorrectness(int width, int height) {
  int matrixSize = width * height;
  double *input = new double[matrixSize]();
  double *output = new double[matrixSize]();

  int lower_bound = 0;
  int upper_bound = 256;

  std::uniform_int_distribution<int> unif(lower_bound, upper_bound);
  std::default_random_engine re;

  for (int i = 0; i < matrixSize; i++) {
    input[i] = unif(re);
    output[i] = 0.0;
  }

  cv::Mat src(height, width, CV_64F, input);
  cv::Mat expected;

  GaussianFilterSlow(input, output, GAUSSIAN_KERNEL_SIZE, width, height,
                     GAUSSIAN_KERNEL_SIGMA);
  cv::GaussianBlur(src, expected,
                   cv::Size(GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE),
                   GAUSSIAN_KERNEL_SIGMA, cv::BORDER_CONSTANT, 0);

  for (int i = 0; i < matrixSize; i++) {
    if (std::abs(output[i] - expected.at<double>(i)) > 1e-6) {
      std::cout << "output[" << i << "] = " << output[i]
                << " expected: " << expected.at<double>(i) << "\n";
      std::cout << "width: " << width << " height: " << height << "\n";
      throw std::runtime_error("TestGaussianFilterSlowCorrectness failed");
    }
  }

  delete[] input;
  delete[] output;
}

void TestGaussianFlterCorrectness(int width, int height) {
  int matrixSize = width * height;
  double *input = new double[matrixSize]();
  double *output = new double[matrixSize]();

  int lower_bound = 0;
  int upper_bound = 256;

  std::uniform_int_distribution<int> unif(lower_bound, upper_bound);
  std::default_random_engine re;

  for (int i = 0; i < matrixSize; i++) {
    input[i] = unif(re);
    output[i] = 0.0;
  }

  cv::Mat src(height, width, CV_64F, input);
  cv::Mat expected;

  GaussianFilter(input, output, GAUSSIAN_KERNEL_SIZE, width, height,
                 GAUSSIAN_KERNEL_SIGMA);

  cv::GaussianBlur(src, expected,
                   cv::Size(GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE),
                   GAUSSIAN_KERNEL_SIGMA, cv::BORDER_CONSTANT, 0);

  for (int i = 0; i < matrixSize; i++) {
    if (std::abs(output[i] - expected.at<double>(i)) > 1e-6) {
      std::cout << "output[" << i << "] = " << output[i]
                << " expected: " << expected.at<double>(i) << "\n";
      std::cout << "width: " << width << " height: " << height << "\n";
      throw std::runtime_error("TestGaussianFilterSlowCorrectness failed");
    }
  }

  delete[] input;
  delete[] output;
}

int main(int argc, char *argv[]) {
  cv::setNumThreads(0);
  try {
    std::cout << "Running tests...\n";
    std::cout << "...Testing matrix padding...\n";
    TestMatrixPadding();

    std::cout << "...Testing GaussianFilterSlow correctness...\n";
    TestGaussianFilterSlowCorrectness(3, 3);
    TestGaussianFilterSlowCorrectness(8, 8);
    TestGaussianFilterSlowCorrectness(32, 32);
    TestGaussianFilterSlowCorrectness(1024, 1024);
    std::cout << "GaussianFilterSlow correctness passed\n";

    std::cout << "...Benchmarking GaussianFilterSlow...\n";
    BenchmarkGaussianFilterSlow(8, 8);
    BenchmarkGaussianFilterSlow(16, 16);
    BenchmarkGaussianFilterSlow(32, 32);
    BenchmarkGaussianFilterSlow(64, 64);

    std::cout << "...Testing GaussianFilter correctness...\n";
    TestGaussianFlterCorrectness(4, 4);
    TestGaussianFlterCorrectness(8, 8);
    TestGaussianFlterCorrectness(32, 32);
    TestGaussianFlterCorrectness(64, 64);
    TestGaussianFlterCorrectness(128, 128);
    TestGaussianFlterCorrectness(256, 256);
    TestGaussianFlterCorrectness(512, 512);
    TestGaussianFlterCorrectness(1024, 1024);
    std::cout << "GaussianFilter correctness passed\n";

    std::cout << "...Benchmarking GaussianFilter...\n";
    BenchmarkGaussianFilter(8, 8);
    BenchmarkGaussianFilter(16, 16);
    BenchmarkGaussianFilter(32, 32);
    BenchmarkGaussianFilter(64, 64);
    BenchmarkGaussianFilter(128, 128);
    BenchmarkGaussianFilter(256, 256);
    BenchmarkGaussianFilter(512, 512);
    BenchmarkGaussianFilter(1024, 1024);

    std::cout << "All tests passed\n";
  } catch (const std::exception &err) {
    std::cerr << "[ERROR] " << err.what() << "\n";

    return -1;
  }

  return 0;
}
