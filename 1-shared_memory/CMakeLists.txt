add_executable(shared_memory shared_memory.cu)
target_link_libraries(shared_memory Eigen3::Eigen basic cublas)