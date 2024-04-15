/**
 * @file basic.h
 * @author qiujiandong <1335521934@qq.com>
 * @date 2024-04-15
 * @brief
 *
 *
 */

#pragma once

/**
 * @brief 基本的矩阵乘法实现。矩阵按列优先存放在一维数组中。
 *
 * @param N - [in] 矩阵维度N
 * @param a - [in] 矩阵a，维度NxN
 * @param b - [in] 矩阵b，维度NxN
 * @param c - [out] 结果矩阵c，维度NxN
 *
 */
__global__ void basic(int N, double *a, double *b, double *c);