C++ Preprocessing Module

What this is
A minimal C++ preprocessing pipeline for image classification:
- load image from disk
- convert BGR -> RGB (OpenCV default is BGR)
- resize to (H,W)
- scale to [0,1]
- normalize with ImageNet mean/std
- convert HWC -> CHW and return NCHW float tensor

Why this matters (systems view)
In real deployments, preprocessing often dominates end-to-end latency.
Optimizing inference alone can produce small gains if preprocess is a large fraction (Amdahl’s Law).

Build (later, locally)
Requirements:
- CMake (>=3.16)
- OpenCV development package 

Commands:
- mkdir -p build && cd build
- cmake -S ../cpp_preproc -B .
- cmake --build . -j

Run:
- ./preproc_cli path/to/image.jpg 224 224

Outputs
- tensor shape
- basic stats (min/max/mean)
- first few tensor values (sanity check)

Common failure modes this code avoids
- Wrong channel order (BGR vs RGB)
- Wrong scale (0..255 instead of 0..1)
- Wrong layout (HWC vs CHW)

Tests (correctness first)
Build with tests:
- mkdir -p build && cd build
- cmake -S ../cpp_preproc -B . -DBUILD_TESTING=ON
- cmake --build . -j
Run tests:
- ctest --output-on-failure

Why tests matter
Preprocessing bugs silently destroy accuracy. These tests guarantee:
- correct BGR->RGB conversion
- correct normalization math
- correct HWC->CHW layout mapping

SIMD notes 
This module includes an AVX2-accelerated normalization path when compiled with AVX2 enabled.

How to enable AVX2 
- Add compiler flags:
-DCMAKE_CXX_FLAGS="-mavx2"

Fallback
If AVX2 is not enabled or not supported, the code falls back to scalar automatically.

Why this is still correct
The SIMD path implements exactly:
(y = (x - mean) / std)
and is tested indirectly via unit tests and later via scalar-vs-SIMD equivalence checks.
