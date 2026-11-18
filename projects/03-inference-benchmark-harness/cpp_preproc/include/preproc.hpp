#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace preproc {

// Output tensor layout: NCHW with N=1, float32
struct Tensor {
  int n = 1;
  int c = 3;
  int h = 224;
  int w = 224;
  std::vector<float> data;  // size = n*c*h*w
};

// ImageNet normalization (standard baseline)
struct NormalizeParams {
  float mean[3] = {0.485f, 0.456f, 0.406f};
  float std[3]  = {0.229f, 0.224f, 0.225f};
};

// Load an image from disk, resize to (out_h, out_w),
// convert to RGB, convert HWC->CHW, normalize, return NCHW tensor (N=1).
Tensor load_resize_normalize(
  const std::string& image_path,
  int out_h,
  int out_w,
  const NormalizeParams& norm = NormalizeParams()
);

// Utility: compute simple stats (min/max/mean) for sanity checking
struct Stats {
  float min_val;
  float max_val;
  float mean_val;
};

Stats tensor_stats(const Tensor& t);

} // namespace preproc
