/**
 * @file coalesce.cu
 * @author qiujiandong <1335521934@qq.com>
 * @date 2024-04-15
 * @brief
 *
 *
 */

#include "coalesce.h"

__global__ void coalesce(double *a, double *b, double *c) {
  // 数据按列存放，所以x按列方向增长，y按行方向增长
  int result_row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int result_col = blockIdx.y * BLOCK_SIZE + threadIdx.y;

  // 将结果清空
  c[result_col * N + result_row] = 0.0;

  // 每个block一起load数据，放入s_a s_b中，同一列的数据在地址上不能对齐，对齐的话会bank访问冲突
  __shared__ double s_a[BLOCK_SIZE][BLOCK_SIZE + 1];
  __shared__ double s_b[BLOCK_SIZE][BLOCK_SIZE + 1];

  // 每个thread需要load N/BLOCK_SIZE次数据
  for (int i = 0; i < N / BLOCK_SIZE; ++i) {
    // (y, x) thread 负责拷贝global中的 (y, x)的数据
    int s_a_row_in_global = result_row;
    int s_a_col_in_global = i * BLOCK_SIZE + threadIdx.y;
    int s_b_col_in_global = result_col;
    int s_b_row_in_global = i * BLOCK_SIZE + threadIdx.x;

    // s_a中的数据需要跨行存放，因为一个warp读的是列数据
    s_a[threadIdx.x][threadIdx.y] = a[s_a_row_in_global + s_a_col_in_global * N];
    s_b[threadIdx.y][threadIdx.x] = b[s_b_row_in_global + s_b_col_in_global * N];
    __syncthreads();

    for (int j = 0; j < BLOCK_SIZE; ++j) {
      c[result_col * N + result_row] += s_a[threadIdx.x][j] * s_b[threadIdx.y][j];
    }
    __syncthreads();
  }
}
