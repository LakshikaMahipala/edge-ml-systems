# toolchains/aarch64-linux-gnu.cmake
# Cross-compilation toolchain for ARM64 (aarch64) Linux

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Toolchain executables (Ubuntu package: gcc-aarch64-linux-gnu, g++-aarch64-linux-gnu)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# Optional: sysroot path (recommended for real device-accurate builds)
# Example (later): set(CMAKE_SYSROOT /path/to/sysroot)
# set(CMAKE_SYSROOT "/opt/sysroots/jetson")

# Ensure CMake searches headers/libs in sysroot first (if set)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Common flags (tune later for target CPU)
set(CMAKE_C_FLAGS_INIT "-O3")
set(CMAKE_CXX_FLAGS_INIT "-O3")
