cmake_minimum_required(VERSION 3.16)
project(cpp_preproc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prefer release builds for benchmarking
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)

add_library(preproc STATIC
  src/preproc.cpp
)

target_include_directories(preproc PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(preproc PUBLIC
  ${OpenCV_LIBS}
)

add_executable(preproc_cli
  src/main.cpp
)

target_link_libraries(preproc_cli PRIVATE
  preproc
)
