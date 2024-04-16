/**
 * @file check_compute_capability.cpp
 * @author qiujiandong <1335521934@qq.com>
 * @date 2024-04-16
 * @brief
 *
 *
 */

#include <iostream>

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Compute Capability: " << prop.major << prop.minor << std::endl;
  return 0;
}
