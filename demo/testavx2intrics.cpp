//
// Created by Meiqi Huang on 24-11-18.
//

#include "testavx2intrics.h"

#include <cstdio>
#include <immintrin.h>  // For AVX2 intrinsics

void vectorized_add_multiply(double *arr1, double *arr2, double *result, int n) {
    int i = 0;
    for (; i <= n - 4; i += 4) {
        // Load 4 values from each array
        __m256d a = _mm256_load_pd(&arr1[i]);
        __m256d b = _mm256_load_pd(&arr2[i]);

        // Perform the computation a * b + a
        __m256d res = _mm256_fmadd_pd(a, b, a);

        // Store the result back to the result array
        _mm256_store_pd(&result[i], res);
    }

    // Handle the remainder if n is not a multiple of 4 (optional)
    for (; i < n; i++) {
        result[i] = arr1[i] * arr2[i] + arr1[i];
    }
}

int main() {
    double arr1[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    double arr2[8] = {8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    double result[8];

    vectorized_add_multiply(arr1, arr2, result, 8);

    for (int i = 0; i < 8; i++) {
        printf("%f ", result[i]);
    }

    return 0;
}
