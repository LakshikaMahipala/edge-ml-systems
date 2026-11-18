cmake_minimum_required(VERSION 3.18)
project(cuda_gemm_optimization LANGUAGES CXX CUDA)

# Default: Turing (T4, sm_75). Override with -DCMAKE_CUDA_ARCHITECTURES=86 etc.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")  # source lines in Nsight Compute

find_package(CUDAToolkit REQUIRED)

add_executable(gemm_bench src/gemm_bench.cu)
target_link_libraries(gemm_bench PRIVATE CUDA::cublas)

add_executable(vector_add src/vector_add.cu)
