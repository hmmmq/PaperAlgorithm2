#pragma once

void GaussianFilter(double *input, double *output, int kernalSize, int width,
                    int height, double sigma);

void GaussianFilterSlow(double *input, double *output, int kernalSize,
                        int width, int height, double sigma);

void GenerateGaussianKernel(double *kernel, int width, int height,
                            double sigma);

void PadMatrix(double *input, double *output, int width, int height,
               int padSize, int padValue);
