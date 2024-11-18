#include <math.h>

void non_max_suppression(double *input, double *output, double *theta,
                         int kernalSize, int width, int height, double sigma) {
    int padd = 1;

    // 遍历每个像素点（排除边缘）
    for (int i = padd; i < height - padd; i++) {
        for (int j = padd; j < width - padd; j++) {
            // 当前像素的索引
            int idx = i * width + j;

            // 转换角度并规范范围为 [0, 180)
            double angle = theta[idx] * 180 / M_PI;
            if (angle < 0) {
                angle += 180;
            }

            double q = 255.0;
            double r = 255.0;

            // 根据梯度方向选择相邻像素
            if ((0 <= angle && angle < 22.5) || (157.5 <= angle && angle <= 180)) {
                q = input[idx + 1];     // 右侧像素
                r = input[idx - 1];     // 左侧像素
            } else if (22.5 <= angle && angle < 67.5) {
                q = input[(i + 1) * width + (j - 1)]; // 左下像素
                r = input[(i - 1) * width + (j + 1)]; // 右上像素
            } else if (67.5 <= angle && angle < 112.5) {
                q = input[(i + 1) * width + j];       // 下方像素
                r = input[(i - 1) * width + j];       // 上方像素
            } else if (112.5 <= angle && angle < 157.5) {
                q = input[(i - 1) * width + (j - 1)]; // 左上像素
                r = input[(i + 1) * width + (j + 1)]; // 右下像素
            }

            // 非极大值抑制
            if (input[idx] >= q && input[idx] >= r) {
                output[idx] = input[idx];
            } else {
                output[idx] = 0.0;
            }
        }
    }
}

#include <immintrin.h>  // For AVX2 intrinsics
#include <cmath>         // For M_PI

void non_max_suppression_optimized(double *input, double *output, double *theta,
                                    int kernalSize, int width, int height, double sigma) {
    int padd = 1;

    // 使用 OpenMP 并行化外层循环，进一步加速
    #pragma omp parallel for collapse(2)
    for (int i = padd; i < height - padd; i++) {
        for (int j = padd; j < width - padd; j += 4) {  // 每次处理 4 个像素
            // 当前像素的索引
            int idx = i * width + j;

            // 使用 SIMD 加载角度数据
            __m256d angle = _mm256_loadu_pd(&theta[idx]);
            angle = _mm256_mul_pd(angle, _mm256_set1_pd(180.0 / M_PI));  // 转换为度

            // 处理负角度，保证角度在 [0, 180) 范围
            __m256d mask = _mm256_cmp_pd(angle, _mm256_setzero_pd(), _CMP_LT_OS);
            angle = _mm256_add_pd(angle, _mm256_and_pd(mask, _mm256_set1_pd(180.0)));

            // 初始化 q 和 r（相邻像素的值）
            __m256d q = _mm256_set1_pd(255.0);
            __m256d r = _mm256_set1_pd(255.0);

            // 根据角度范围选择相邻像素
            __m256d mask1 = _mm256_and_pd(_mm256_cmp_pd(angle, _mm256_set1_pd(22.5), _CMP_GE_OS), _mm256_cmp_pd(angle, _mm256_set1_pd(157.5), _CMP_LT_OS));
            __m256d mask2 = _mm256_and_pd(_mm256_cmp_pd(angle, _mm256_set1_pd(22.5), _CMP_GE_OS), _mm256_cmp_pd(angle, _mm256_set1_pd(67.5), _CMP_LT_OS));
            __m256d mask3 = _mm256_and_pd(_mm256_cmp_pd(angle, _mm256_set1_pd(67.5), _CMP_GE_OS), _mm256_cmp_pd(angle, _mm256_set1_pd(112.5), _CMP_LT_OS));
            __m256d mask4 = _mm256_and_pd(_mm256_cmp_pd(angle, _mm256_set1_pd(112.5), _CMP_GE_OS), _mm256_cmp_pd(angle, _mm256_set1_pd(157.5), _CMP_LT_OS));

            // 根据角度范围来选择不同的相邻像素
            __m256d q_masked = _mm256_blendv_pd(q, _mm256_loadu_pd(&input[idx + 1]), mask1); // 右侧像素
            __m256d r_masked = _mm256_blendv_pd(r, _mm256_loadu_pd(&input[idx - 1]), mask1); // 左侧像素

            // 左下与右上
            q_masked = _mm256_blendv_pd(q_masked, _mm256_loadu_pd(&input[(i + 1) * width + (j - 1)]), mask2);
            r_masked = _mm256_blendv_pd(r_masked, _mm256_loadu_pd(&input[(i - 1) * width + (j + 1)]), mask2);

            // 下方与上方
            q_masked = _mm256_blendv_pd(q_masked, _mm256_loadu_pd(&input[(i + 1) * width + j]), mask3);
            r_masked = _mm256_blendv_pd(r_masked, _mm256_loadu_pd(&input[(i - 1) * width + j]), mask3);

            // 左上与右下
            q_masked = _mm256_blendv_pd(q_masked, _mm256_loadu_pd(&input[(i - 1) * width + (j - 1)]), mask4);
            r_masked = _mm256_blendv_pd(r_masked, _mm256_loadu_pd(&input[(i + 1) * width + (j + 1)]), mask4);

            // 使用 SIMD 对当前像素与相邻像素进行比较，选择最大的值
            __m256d input_val = _mm256_loadu_pd(&input[idx]);
            __m256d result = _mm256_max_pd(_mm256_max_pd(input_val, q_masked), r_masked);

            // 非极大值抑制：如果当前值小于相邻值，则设置为 0
            __m256d output_val = _mm256_blendv_pd(_mm256_setzero_pd(), result, _mm256_cmp_pd(input_val, result, _CMP_GE_OS));

            // 将结果存回输出数组
            _mm256_storeu_pd(&output[idx], output_val);
        }
    }
}
