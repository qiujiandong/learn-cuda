/**
 * @file basic.cu
 * @author qiujiandong <1335521934@qq.com>
 * @date 2024-04-15
 * @brief
 *
 *
 */

#include "basic.h"

__global__ void basic(int N, double *a, double *b, double *c) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  double sum = 0.0;
  for (int i = 0; i < N; i++) {
    sum += a[row + i * N] * b[col * N + i];
  }
  c[col * N + row] = sum;
}