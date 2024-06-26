/**
 * @file other_practice.cu
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
#include "coalesce.h"
#include "common.h"

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

  // 准备用graph实现
  Eigen::MatrixXd result_graph = Eigen::MatrixXd::Zero(N, N);
  cudaStream_t s;
  cudaStreamCreate(&s);
  cudaHostRegister(mat1.data(), N * N * sizeof(double), cudaHostRegisterDefault);
  cudaHostRegister(mat2.data(), N * N * sizeof(double), cudaHostRegisterDefault);
  cudaHostRegister(result_graph.data(), N * N * sizeof(double), cudaHostRegisterDefault);

  // 创建graph和exec
  cudaGraph_t graph;
  cudaGraphCreate(&graph, 0);
  cudaGraphExec_t exec;

  // 创建capture graph
  cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed);
  cudaMemcpyAsync(d_mat1, mat1.data(), N * N * sizeof(double), cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(d_mat2, mat2.data(), N * N * sizeof(double), cudaMemcpyHostToDevice, s);
  coalesce<<<gridSize, blockSize, 0, s>>>(d_mat1, d_mat2, d_result);
  cudaMemcpyAsync(result_graph.data(), d_result, N * N * sizeof(double), cudaMemcpyDeviceToHost, s);
  cudaStreamEndCapture(s, &graph);

  // graph实例化
  cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);

  // 运行graph
  start = std::chrono::high_resolution_clock::now();
  cudaGraphLaunch(exec, s);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();

  // 检查结果
  if (!result_graph.isApprox(result_cuda, 1e-6)) {
    std::cout << "Result mismatch" << std::endl;
    return -1;
  }

  // 统计用时
  duration = end - start;
  std::cout << "Cuda graph matrix multiplication time: " << duration.count() << " seconds" << std::endl;

  // 准备用memory map的方式实现
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  if (prop.canMapHostMemory != 1) {
    std::cout << "Device cannot map host memory" << std::endl;
    return -1;
  }
  cudaSetDeviceFlags(cudaDeviceMapHost);

  // 分配结果存放的空间，获取map后的device端地址
  Eigen::MatrixXd reuslt_mapped = Eigen::MatrixXd::Zero(N, N);
  double *d_mat1_mapped, *d_mat2_mapped, *d_result_mapped;
  cudaHostRegister(reuslt_mapped.data(), N * N * sizeof(double), cudaHostRegisterDefault);
  cudaHostGetDevicePointer(&d_mat1_mapped, mat1.data(), 0);
  cudaHostGetDevicePointer(&d_mat2_mapped, mat2.data(), 0);
  cudaHostGetDevicePointer(&d_result_mapped, reuslt_mapped.data(), 0);

  // 进行矩阵乘法
  start = std::chrono::high_resolution_clock::now();
  coalesce<<<gridSize, blockSize>>>(d_mat1_mapped, d_mat2_mapped, d_result_mapped);
  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();

  // 检查结果
  if (!reuslt_mapped.isApprox(result_cuda, 1e-6)) {
    std::cout << "Result mismatch" << std::endl;
    return -1;
  }

  // 统计用时
  duration = end - start;
  std::cout << "Cuda mapped matrix multiplication time: " << duration.count() << " seconds" << std::endl;
}