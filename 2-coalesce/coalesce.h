/**
 * @file coalesce.h
 * @author qiujiandong <1335521934@qq.com>
 * @date 2024-04-15
 * @brief
 *
 *
 */

#pragma once

#include "common.h"

/**
 * @brief 利用shared memory实现的矩阵乘法，同时考虑coalesce访问
 *
 * @param a - [in] 矩阵a，维度NxN
 * @param b - [in] 矩阵b，维度NxN
 * @param c - [out] 结果矩阵c，维度NxN，返回a*b
 */
__global__ void coalesce(double *a, double *b, double *c);