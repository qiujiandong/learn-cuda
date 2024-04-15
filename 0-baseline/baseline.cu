/**
 * @file baseline.cu
 * @author qiujiandong <1335521934@qq.com>
 * @date 2024-04-15
 * @brief
 *
 *
 */

#include <cublas_v2.h>

#include <Eigen/Dense>
#include <chrono>
#include <iostream>

#include "basic.h"
#include "common.h"

int main() {
  // 创建两个NxN的随机矩阵，用于进行矩阵乘法
  Eigen::MatrixXd mat1 = Eigen::MatrixXd::Random(N, N);
  Eigen::MatrixXd mat2 = Eigen::MatrixXd::Random(N, N);

  // 用Eigen实现矩阵乘法
  Eigen::MatrixXd result_eigen = Eigen::MatrixXd::Zero(N, N);
  auto start = std::chrono::high_resolution_clock::now();
  result_eigen = mat1 * mat2;
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "Eigen matrix multiplication time: " << duration.count() << " seconds" << std::endl;

  // 在设备上分配空间
  double *d_mat1, *d_mat2, *d_result;
  cudaMalloc(&d_mat1, N * N * sizeof(double));
  cudaMalloc(&d_mat2, N * N * sizeof(double));
  cudaMalloc(&d_result, N * N * sizeof(double));

  // 设置block和grid
  dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridSize(N / blockSize.x, N / blockSize.y);

  // 进行矩阵乘法。将数据拷贝到设备，再将结果拷贝回来
  Eigen::MatrixXd result_cuda = Eigen::MatrixXd::Zero(N, N);
  start = std::chrono::high_resolution_clock::now();
  cudaMemcpy(d_mat1, mat1.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mat2, mat2.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);
  basic<<<gridSize, blockSize>>>(N, d_mat1, d_mat2, d_result);
  cudaMemcpy(result_cuda.data(), d_result, N * N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();

  // 检查结果是否正确
  if (!result_cuda.isApprox(result_eigen, 1e-6)) {
    std::cout << "Result mismatch" << std::endl;
    return -1;
  }

  // 统计用时
  duration = end - start;
  std::cout << "Cuda matrix multiplication time: " << duration.count() << " seconds" << std::endl;

  // 准备用cublas实现矩阵乘法
  Eigen::MatrixXd result_cublas = Eigen::MatrixXd::Zero(N, N);
  cudaHostRegister(result_cublas.data(), N * N * sizeof(double), cudaHostRegisterDefault);
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  cudaStream_t s;
  cudaStreamCreate(&s);
  cublasSetStream(cublas_handle, s);
  double alpha = 1.0;
  double beta = 0.0;

  // 调用cublas进行运算
  start = std::chrono::high_resolution_clock::now();
  cudaMemcpyAsync(d_mat1, mat1.data(), N * N * sizeof(double), cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(d_mat2, mat2.data(), N * N * sizeof(double), cudaMemcpyHostToDevice, s);
  cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_mat1, N, d_mat2, N, &beta, d_result, N);
  cudaMemcpyAsync(result_cublas.data(), d_result, N * N * sizeof(double), cudaMemcpyDeviceToHost, s);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();

  // 检查结果
  if (!result_cublas.isApprox(result_eigen, 1e-6)) {
    std::cout << "Result mismatch" << std::endl;
    return -1;
  }

  // 统计用时
  duration = end - start;
  std::cout << "Cublas matrix multiplication time: " << duration.count() << " seconds" << std::endl;

  cublasDestroy(cublas_handle);
  cudaStreamDestroy(s);

  // 释放在设备上分配的空间
  cudaFree(d_mat1);
  cudaFree(d_mat2);
  cudaFree(d_result);

  return 0;
}