#ifndef DOUBLE_THRESHOLD_H
#define DOUBLE_THRESHOLD_H

#endif //DOUBLE_THRESHOLD_H
#pragma once

void DoubleThreshold(double *input, double *output, int width, int height,
                     double lowThreshold, double highThreshold);

void DoubleThresholdSlow(double *input, double *output, int height, int width,
                     double lowThreshold, double highThreshold);