# toolchains/armv7-linux-gnueabihf.cmake
# Cross-compilation toolchain for ARMv7 (32-bit) Linux hard-float

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

# Toolchain executables (Ubuntu package: gcc-arm-linux-gnueabihf, g++-arm-linux-gnueabihf)
set(CMAKE_C_COMPILER arm-linux-gnueabihf-gcc)
set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)

# Optional sysroot (strongly recommended)
# set(CMAKE_SYSROOT "/opt/sysroots/rpi32")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Conservative baseline flags
set(CMAKE_C_FLAGS_INIT "-O3")
set(CMAKE_CXX_FLAGS_INIT "-O3")
