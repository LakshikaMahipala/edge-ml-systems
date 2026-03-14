#include "preproc.hpp"
#include "simd.hpp"

#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <numeric>

namespace preproc {

static inline size_t idx_nchw(int n, int c, int h, int w, int C, int H, int W) {
  return static_cast<size_t>(n) * C * H * W
       + static_cast<size_t>(c) * H * W
       + static_cast<size_t>(h) * W
       + static_cast<size_t>(w);
}

Tensor load_resize_normalize(
  const std::string& image_path,
  int out_h,
  int out_w,
  const NormalizeParams& norm
) {
  // OpenCV loads BGR by default
  cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
  if (bgr.empty()) {
    throw std::runtime_error("Failed to read image: " + image_path);
  }

  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

  cv::Mat resized;
  cv::resize(rgb, resized, cv::Size(out_w, out_h), 0, 0, cv::INTER_LINEAR);

  // Convert to float32 in range [0,1]
  cv::Mat f32;
  resized.convertTo(f32, CV_32FC3, 1.0 / 255.0);

  Tensor t;
  t.n = 1;
  t.c = 3;
  t.h = out_h;
  t.w = out_w;
  t.data.resize(static_cast<size_t>(t.n) * t.c * t.h * t.w);

  // HWC -> CHW + normalize
  "
  for (int y = 0; y < out_h; ++y) {
    const cv::Vec3f* row = f32.ptr<cv::Vec3f>(y);
    for (int x = 0; x < out_w; ++x) {
      const cv::Vec3f px = row[x]; // RGB in [0,1]
      for (int c = 0; c < 3; ++c) {
        float v = (px[c] - norm.mean[c]) / norm.std[c];
        t.data[idx_nchw(0, c, y, x, t.c, t.h, t.w)] = v;
      }
    }
  }
  "
  // f32 is CV_32FC3 (RGB, HWC)
  // We will call SIMD-aware conversion+normalize into CHW.
  const float* hwc_ptr = reinterpret_cast<const float*>(f32.data);

  preproc::hwc_rgb_to_chw_normalize(
      hwc_ptr,
      out_h,
      out_w,
      norm.mean,
      norm.std,
      t.data.data()
  );

  return t;
}

Stats tensor_stats(const Tensor& t) {
  if (t.data.empty()) {
    return Stats{0.f, 0.f, 0.f};
  }

  float mn = t.data[0];
  float mx = t.data[0];
  double sum = 0.0;

  for (float v : t.data) {
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    sum += static_cast<double>(v);
  }

  float mean = static_cast<float>(sum / static_cast<double>(t.data.size()));
  return Stats{mn, mx, mean};
}

} // namespace preproc
