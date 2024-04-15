cmake_minimum_required(VERSION 3.0)

set(CMAKE_CUDA_ARCHITECTURES 70 72 75 80 86 89)
project(learn-cuda LANGUAGES CXX CUDA)

find_package(Eigen3 REQUIRED)

# 每个cpu thread都有一个独立的default stream
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")

# 显示详细的 ptx 信息
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

add_subdirectory(0-baseline)

add_subdirectory(1-shared_memory)

add_subdirectory(2-coalesce)

add_subdirectory(3-other_practice)