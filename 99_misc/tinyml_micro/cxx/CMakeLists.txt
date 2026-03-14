cmake_minimum_required(VERSION 3.16)
project(tinyml_micro CXX)

set(CMAKE_CXX_STANDARD 17)
add_executable(tinyml_micro
  main.cpp
  micro_infer.cpp
)

target_include_directories(tinyml_micro PRIVATE .)
