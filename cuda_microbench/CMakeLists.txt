cmake_minimum_required(VERSION 3.18)
project(cuda_microbench LANGUAGES CXX CUDA)

add_executable(vector_add vector_add.cu)
set_target_properties(vector_add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

cmake_minimum_required(VERSION 3.18)
project(cuda_microbench LANGUAGES CXX CUDA)

add_executable(vector_add vector_add.cu)
set_target_properties(vector_add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(gemm_naive gemm_naive.cu)
set_target_properties(gemm_naive PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(gemm_cublas gemm_cublas.cu)
set_target_properties(gemm_cublas PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link cuBLAS
find_package(CUDAToolkit REQUIRED)
target_link_libraries(gemm_cublas PRIVATE CUDA::cublas)

add_executable(gemm_tiled gemm_tiled.cu)
set_target_properties(gemm_tiled PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
