#ifndef NON_MAX_SUPPRESSION_H
#define NON_MAX_SUPPRESSION_H

#include <cmath> // 用于 M_PI

/**
 * @brief 非极大值抑制函数，优化图像边缘检测。
 *
 * @param input 输入图像的像素数组，按行优先顺序存储。
 * @param output 输出图像的像素数组，与输入图像大小一致。
 * @param theta 梯度方向数组，按行优先顺序存储。
 * @param kernalSize 内核大小（当前未使用，可扩展）。
 * @param width 图像的宽度。
 * @param height 图像的高度。
 * @param sigma 标准差参数（当前未使用，可扩展）。
 */
void non_max_suppression(double *input, double *output, double *theta,
                         int kernalSize, int width, int height, double sigma);
void non_max_suppression_optimized(double *input, double *output, double *theta,
                                    int kernalSize, int width, int height, double sigma);
#endif // NON_MAX_SUPPRESSION_H
