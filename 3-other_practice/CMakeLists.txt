add_executable(other_practice other_practice.cu)
target_link_libraries(other_practice Eigen3::Eigen basic coalesce cublas)