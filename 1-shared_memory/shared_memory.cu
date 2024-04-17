/**
 * @file shared_memory.cu
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

__global__ void shared_memory(double *a, double *b, double *c) {
  int result_row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  int result_col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  // 将结果清空
  c[result_row + result_col * N] = 0.0;

  int a_col_global, b_row_global;

  // 每个block一起load数据，放入s_a s_b中
  __shared__ double s_a[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ double s_b[BLOCK_SIZE][BLOCK_SIZE];

  // 每个thread需要load N/BLOCK_SIZE次数据
  for (int i = 0; i < N / BLOCK_SIZE; ++i) {
    // 计算要搬运的数据在global下的索引
    a_col_global = i * BLOCK_SIZE + threadIdx.x;
    b_row_global = i * BLOCK_SIZE + threadIdx.y;

    // 搬运数据
    s_a[threadIdx.y][threadIdx.x] = a[result_row + a_col_global * N];
    s_b[threadIdx.y][threadIdx.x] = b[b_row_global + result_col * N];
    __syncthreads();

    // 计算部分和
    for (int j = 0; j < BLOCK_SIZE; ++j) {
      c[result_row + result_col * N] += s_a[threadIdx.y][j] * s_b[j][threadIdx.x];
    }
    __syncthreads();
  }
}

int main() {
  // 创建两个NxN的随机矩阵，用于进行矩阵乘法
  Eigen::MatrixXd mat1 = Eigen::MatrixXd::Random(N, N);
  Eigen::MatrixXd mat2 = Eigen::MatrixXd::Random(N, N);

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
  auto start = std::chrono::high_resolution_clock::now();
  cudaMemcpy(d_mat1, mat1.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mat2, mat2.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);
  basic<<<gridSize, blockSize>>>(N, d_mat1, d_mat2, d_result);
  cudaMemcpy(result_cuda.data(), d_result, N * N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  // 统计用时
  std::chrono::duration<double> duration = end - start;
  std::cout << "Cuda matrix multiplication time: " << duration.count() << " seconds" << std::endl;

  Eigen::MatrixXd result_shmem = Eigen::MatrixXd::Zero(N, N);
  cudaStream_t s;
  cudaStreamCreate(&s);
  cudaHostRegister(mat1.data(), N * N * sizeof(double), cudaHostRegisterDefault);
  cudaHostRegister(mat2.data(), N * N * sizeof(double), cudaHostRegisterDefault);
  cudaHostRegister(result_shmem.data(), N * N * sizeof(double), cudaHostRegisterDefault);

  // 异步搬运数据，用共享内存实现矩阵乘法
  start = std::chrono::high_resolution_clock::now();
  cudaMemcpyAsync(d_mat1, mat1.data(), N * N * sizeof(double), cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(d_mat2, mat2.data(), N * N * sizeof(double), cudaMemcpyHostToDevice, s);
  shared_memory<<<gridSize, blockSize, 0, s>>>(d_mat1, d_mat2, d_result);
  cudaMemcpyAsync(result_shmem.data(), d_result, N * N * sizeof(double), cudaMemcpyDeviceToHost, s);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();

  // 检查结果
  if (!result_shmem.isApprox(result_cuda, 1e-6)) {
    std::cout << "Result mismatch" << std::endl;
    return -1;
  }

  // 统计用时
  duration = end - start;
  std::cout << "Shared memory matrix multiplication time: " << duration.count() << " seconds" << std::endl;

  // 准备用cublas实现矩阵乘法
  Eigen::MatrixXd result_cublas = Eigen::MatrixXd::Zero(N, N);
  cudaHostRegister(result_cublas.data(), N * N * sizeof(double), cudaHostRegisterDefault);
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
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
  if (!result_cublas.isApprox(result_cuda, 1e-6)) {
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