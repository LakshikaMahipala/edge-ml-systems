#pragma once

#include <cstddef>
#include <cstdint>

namespace preproc {

// Returns true if compiled with AVX2 enabled (compile-time).
bool simd_enabled();

// Normalize and write NCHW output.
// Input is interleaved RGB float array in HWC order (OpenCV CV_32FC3).
// Output is float NCHW buffer (size = 3*H*W).
//
// This function will use SIMD if available; otherwise scalar.
void hwc_rgb_to_chw_normalize(
    const float* hwc_rgb, // length = H*W*3
    int H,
    int W,
    const float mean[3],
    const float stdv[3],
    float* out_chw // length = 3*H*W
);

} // namespace preproc
