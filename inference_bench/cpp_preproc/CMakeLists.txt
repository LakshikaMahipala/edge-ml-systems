cmake_minimum_required(VERSION 3.16)
project(cpp_preproc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)

add_library(preproc STATIC
  src/preproc.cpp
  src/simd.cpp
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

# =========================
# Tests (Catch2 via FetchContent)
# =========================
include(CTest)

if(BUILD_TESTING)
  include(FetchContent)

  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.4
  )
  FetchContent_MakeAvailable(Catch2)

  add_executable(test_preproc
    tests/test_preproc.cpp
  )

  target_link_libraries(test_preproc PRIVATE
    preproc
    Catch2::Catch2WithMain
  )

  add_test(NAME test_preproc COMMAND test_preproc)
endif()


# =========================
# Python extension (pybind11)
# =========================
include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.12.0
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(cpp_preproc_ext
  python/bindings.cpp
)

target_link_libraries(cpp_preproc_ext PRIVATE
  preproc
)
