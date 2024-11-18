#ifndef DOUBLE_THRESHOLD_H
#define DOUBLE_THRESHOLD_H

// 双阈值函数声明
// 输入参数:
// - double *input: 输入图像的像素数组，按行优先存储
// - double *output: 输出图像的像素数组，与输入大小一致
// - int kernalSize: 内核大小（当前未使用，可为扩展保留）
// - int width: 图像的宽度
// - int height: 图像的高度
// - double sigma: 标准差参数（当前未使用，可为扩展保留）
// - double low_thres: 较低的阈值，默认值为 50
// - double high_thres: 较高的阈值，默认值为 100
void double_threshold(double *input, double *output, int kernalSize,
                      int width, int height, double sigma,
                      double low_thres = 50, double high_thres = 100);
// double_threshold 优化版本声明
void double_threshold_optimized(double *input, double *output, int width, int height,
                                 double low_thres = 50, double high_thres = 100);
#endif // DOUBLE_THRESHOLD_H

