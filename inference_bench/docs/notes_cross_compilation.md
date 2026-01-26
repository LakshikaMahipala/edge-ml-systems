Cross-compilation Notes 

Core definitions
- Host: machine performing compilation (often x86_64 laptop)
- Target: machine that will run the binary (often ARM embedded Linux)
- Toolchain: compiler + linker + binutils that emit target machine code
- Sysroot: directory that contains target headers + libraries (the target “/usr”, “/lib”, etc.)

Why sysroot matters
Cross compiling without a sysroot often fails because dependencies (OpenCV, libc, etc.) differ by CPU/OS.
Correct builds require:
- target headers (include)
- target libraries (lib)
- correct linker paths

What we added
- toolchains/aarch64-linux-gnu.cmake
- toolchains/armv7-linux-gnueabihf.cmake

Later: typical build command (Linux host)
Example for cpp_preproc:
- cmake -S cpp_preproc -B build_aarch64 -DCMAKE_TOOLCHAIN_FILE=toolchains/aarch64-linux-gnu.cmake
- cmake --build build_aarch64 -j

If using sysroot (recommended)
- cmake ... -DCMAKE_TOOLCHAIN_FILE=... -DCMAKE_SYSROOT=/path/to/sysroot

Where sysroot comes from (practical options)
1) Copy from target device:
   - rsync /lib /usr/include /usr/lib (and related) into a local folder
2) Use vendor-provided sysroot (Jetson SDK / Yocto SDK / Buildroot SDK)
3) Build inside a target-like container (sometimes easier than cross compile)

Common failure modes
- Found host OpenCV instead of target OpenCV (wrong headers/libs)
- Linker errors: “wrong ELF class” (mixing ARM and x86 libraries)
- Python extension cross-compiling (pybind11) is harder: you need target Python headers/libs too.
  For this project, we focus on cross-compiling the C++ library first, then handle Python binding later.

How this fits the repo timeline
- Week 3 Day 2: add toolchain skeleton
- Week 3 Day 3: containerize builds (Docker)
- Week 3 Day 4+: run ONNX Runtime comparisons
