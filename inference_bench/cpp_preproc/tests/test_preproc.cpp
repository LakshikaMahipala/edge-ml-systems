#include "preproc.hpp"

#include <catch2/catch_all.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <stdexcept>

// We will test core math and layout correctness.
// To avoid filesystem dependencies, we create an image in memory,
// write it to a temporary file, then load via existing API.
// (Later we can extend API to accept cv::Mat directly.)

static std::string write_temp_png(const cv::Mat& bgr) {
  // Use a stable filename (overwrites). In real CI, this is fine.
  const std::string path = "tmp_test_image.png";
  if (!cv::imwrite(path, bgr)) {
    throw std::runtime_error("Failed to write temp image");
  }
  return path;
}

static float expected_norm(float x01, float mean, float std) {
  return (x01 - mean) / std;
}

TEST_CASE("Preproc produces correct shape and size", "[preproc]") {
  // Create a 2x2 BGR image.
  cv::Mat bgr(2, 2, CV_8UC3, cv::Scalar(0, 0, 0));
  auto path = write_temp_png(bgr);

  preproc::Tensor t = preproc::load_resize_normalize(path, 2, 2);

  REQUIRE(t.n == 1);
  REQUIRE(t.c == 3);
  REQUIRE(t.h == 2);
  REQUIRE(t.w == 2);
  REQUIRE(t.data.size() == static_cast<size_t>(1 * 3 * 2 * 2));
}

TEST_CASE("BGR->RGB conversion is correct", "[preproc]") {
  // We craft one pixel with distinct channels in BGR.
  // OpenCV loads BGR. Our code converts to RGB.
  //
  // Let pixel be B=10, G=20, R=30 in uint8.
  // After conversion to RGB, channels should be [R,G,B] = [30,20,10].
  // After scaling to [0,1], values are /255.
  // Then normalized with ImageNet mean/std.

  cv::Mat bgr(1, 1, CV_8UC3);
  bgr.at<cv::Vec3b>(0, 0) = cv::Vec3b(10, 20, 30); // B,G,R
  auto path = write_temp_png(bgr);

  preproc::NormalizeParams norm;
  preproc::Tensor t = preproc::load_resize_normalize(path, 1, 1, norm);

  // Tensor is NCHW with N=1, C=3, H=1, W=1
  // Index mapping: data[c * H * W + y * W + x] since N=1
  float r01 = 30.0f / 255.0f;
  float g01 = 20.0f / 255.0f;
  float b01 = 10.0f / 255.0f;

  float exp_r = expected_norm(r01, norm.mean[0], norm.std[0]);
  float exp_g = expected_norm(g01, norm.mean[1], norm.std[1]);
  float exp_b = expected_norm(b01, norm.mean[2], norm.std[2]);

  // Allow small floating error
  REQUIRE(t.data[0] == Catch::Approx(exp_r).epsilon(1e-5)); // channel 0 (R)
  REQUIRE(t.data[1] == Catch::Approx(exp_g).epsilon(1e-5)); // channel 1 (G)
  REQUIRE(t.data[2] == Catch::Approx(exp_b).epsilon(1e-5)); // channel 2 (B)
}

TEST_CASE("HWC->CHW layout mapping is correct for 1x2 image", "[preproc]") {
  // Make a 1x2 image with two different pixels to verify spatial indexing.
  // Pixel0: B=0,G=0,R=255  => RGB = [255,0,0]
  // Pixel1: B=0,G=255,R=0  => RGB = [0,255,0]
  cv::Mat bgr(1, 2, CV_8UC3);
  bgr.at<cv::Vec3b>(0, 0) = cv::Vec3b(0, 0, 255);
  bgr.at<cv::Vec3b>(0, 1) = cv::Vec3b(0, 255, 0);

  auto path = write_temp_png(bgr);

  preproc::NormalizeParams norm;
  preproc::Tensor t = preproc::load_resize_normalize(path, 1, 2, norm);

  // For H=1, W=2, the CHW layout is:
  // R channel: [x0, x1]
  // G channel: [x0, x1]
  // B channel: [x0, x1]
  auto idx = [&](int c, int x) { return c * (1 * 2) + 0 * 2 + x; };

  // Pixel0 RGB = [1,0,0]
  float r0 = expected_norm(1.0f, norm.mean[0], norm.std[0]);
  float g0 = expected_norm(0.0f, norm.mean[1], norm.std[1]);
  float b0 = expected_norm(0.0f, norm.mean[2], norm.std[2]);

  // Pixel1 RGB = [0,1,0]
  float r1 = expected_norm(0.0f, norm.mean[0], norm.std[0]);
  float g1 = expected_norm(1.0f, norm.mean[1], norm.std[1]);
  float b1 = expected_norm(0.0f, norm.mean[2], norm.std[2]);

  REQUIRE(t.data[idx(0, 0)] == Catch::Approx(r0).epsilon(1e-5));
  REQUIRE(t.data[idx(1, 0)] == Catch::Approx(g0).epsilon(1e-5));
  REQUIRE(t.data[idx(2, 0)] == Catch::Approx(b0).epsilon(1e-5));

  REQUIRE(t.data[idx(0, 1)] == Catch::Approx(r1).epsilon(1e-5));
  REQUIRE(t.data[idx(1, 1)] == Catch::Approx(g1).epsilon(1e-5));
  REQUIRE(t.data[idx(2, 1)] == Catch::Approx(b1).epsilon(1e-5));
}
