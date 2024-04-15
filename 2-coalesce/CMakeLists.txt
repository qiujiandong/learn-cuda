add_library(coalesce coalesce.cu)
target_include_directories(coalesce PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(run_coalesce main.cu)
target_link_libraries(run_coalesce Eigen3::Eigen basic coalesce cublas)