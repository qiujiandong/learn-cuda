add_library(basic basic.cu)
target_include_directories(basic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(baseline baseline.cu)
target_link_libraries(baseline basic Eigen3::Eigen cublas)
