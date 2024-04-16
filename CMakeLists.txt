cmake_minimum_required(VERSION 3.0)

# 根据需要修改/添加架构的配置
set(CMAKE_CUDA_ARCHITECTURES 75 86)
project(learn-cuda LANGUAGES CXX CUDA)

find_package(Eigen3 REQUIRED)

# 每个cpu thread都有一个独立的default stream
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")

# 显示详细的 ptx 信息
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

# 为了include common.h 定义了矩阵的size和block的size
include_directories(${CMAKE_SOURCE_DIR})

# 检查显卡的Compute Capability
add_executable(check_cc check_compute_capability.cu)
add_custom_command(TARGET check_cc POST_BUILD
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/check_cc
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# 最基础的矩阵乘法实现方式与eigen，cublas的实现进行对比
add_subdirectory(0-baseline)

# 用共享内存实现，减少访问global memory的次数
add_subdirectory(1-shared_memory)

# 尽可能用coalesce的形式访存；减少访问shared memory的bank冲突
add_subdirectory(2-coalesce)

# 用capture graph执行矩阵乘法；用memory mapped方式执行矩阵乘法
add_subdirectory(3-other_practice)
