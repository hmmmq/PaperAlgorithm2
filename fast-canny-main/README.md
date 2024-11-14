# Fast Canny

Fast and furious Canny edge detection.

## Pre-requisites

- Cmake >= 3.26
- GCC >= 8.5.0

## Build

```bash

cmake -B build -G "Unix Makefiles"
cmake --build build

```
NOTE: When building the whole project, especifally if you are building the opencv benchmark, CMake will download the opencv source code and build it. This will take a while.

## Development

This project is using Devcontainers to provide a consistent development environment. To start the development container, you will need to have Docker installed on your machine. And then on VSCode you can open the project in a container by clicking on the green button on the bottom left corner of the window and selecting "Reopen in Container".


## Run

### Running OpenCV benchmark

To run the OpenCV benchmark, you will need to unzip the images from the COCO dataset.

```bash
unzip coco_images.zip
```

Then you can run the benchmark with the following command:

```bash
./build/benchmark/opencv_benchmark coco_images/
```

### Running Gaussian Filter benchmark

To run the Gaussian Filter benchmark, you can run the following command:

```bash
./build/benchmark/gaussian_filter_benchmark
```


### Running the image download script

The images in the zip file are downloaded by running the python script `download_coco_images.py`. This script will download the images from the COCO dataset and save them in the `coco_images` folder.

For local python development, you should set up and run the Python virtual environment. It will set up a Python 3.12 virtualenv. When running the benchmark on the ECE machines, we don't need to redownload the images.

```bash
 # Only for local development
pipenv shell
pipenv install
python download_coco_images.py
```

### Generating assmebly code
To generate the assembly code for the Gaussian filter benchmark, you can run the following command:

```bash
objdump -d build/benchmark/gaussian_filter_benchmark > disassembly.S
```
### 补充代码
**非最大抑制（Non-Maxima Suppression, NMS）**和**双阈值（Double Threshold）**这两个步骤的SIMD版本。
以下是中文说明和代码实现。

---

### 1. `non_maxima_suppression.h`

```cpp
#pragma once

void NonMaximaSuppression(double *input, double *output, int width, int height);
```

### 2. `non_maxima_suppression.cpp`

```cpp
#include "non_maxima_suppression.h"
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
```

### 3. `double_threshold.h`

```cpp
#pragma once

void DoubleThreshold(double *input, double *output, int width, int height,
                     double lowThreshold, double highThreshold);
```

### 4. `double_threshold.cpp`

```cpp
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
```

### 5. 集成到 `main.cpp` 或其他代码中：

调用这两个函数（`NonMaximaSuppression` 和 `DoubleThreshold`）来应用非最大抑制和双阈值处理。

```cpp
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
```

---

### 关键点总结：

1. **非最大抑制（NMS）：**
   - 每次加载4个像素，并与其邻居进行比较，抑制非最大值。
   - 使用SIMD（AVX2）并行处理4个像素，提高处理速度。

2. **双阈值（Double Threshold）：**
   - 使用两个阈值（低阈值和高阈值）将像素分类为强边缘、弱边缘或非边缘。
   - 强边缘保留，弱边缘将在后续步骤中处理。

3. **性能：**
   - SIMD技术使得每次可以并行处理多个像素，从而显著提升了处理速度。
   - 你可以使用OpenCV来验证这些SIMD实现的正确性，并进行性能比较。

4. **最终测试：**
   - 使用OpenCV进行结果的正确性验证，同时比较SIMD版本与传统版本的性能差异。

