add_library(basic basic.cu)

add_executable(baseline baseline.cu)
target_link_libraries(baseline basic Eigen3::Eigen cublas)