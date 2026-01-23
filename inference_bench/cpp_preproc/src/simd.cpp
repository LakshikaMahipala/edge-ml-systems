#include "simd.hpp"

#include <cstring>

#if defined(__AVX2__)
  #include <immintrin.h>
#endif

namespace preproc {

bool simd_enabled() {
#if defined(__AVX2__)
  return true;
#else
  return false;
#endif
}

// Scalar reference (trusted)
static void hwc_rgb_to_chw_normalize_scalar(
    const float* hwc_rgb, int H, int W,
    const float mean[3], const float stdv[3],
    float* out_chw
) {
  const int HW = H * W;
  float* outR = out_chw + 0 * HW;
  float* outG = out_chw + 1 * HW;
  float* outB = out_chw + 2 * HW;

  for (int i = 0; i < HW; ++i) {
    float r = hwc_rgb[3*i + 0];
    float g = hwc_rgb[3*i + 1];
    float b = hwc_rgb[3*i + 2];

    outR[i] = (r - mean[0]) / stdv[0];
    outG[i] = (g - mean[1]) / stdv[1];
    outB[i] = (b - mean[2]) / stdv[2];
  }
}

#if defined(__AVX2__)
// AVX2 optimized path
static void hwc_rgb_to_chw_normalize_avx2(
    const float* hwc_rgb, int H, int W,
    const float mean[3], const float stdv[3],
    float* out_chw
) {
  const int HW = H * W;
  float* outR = out_chw + 0 * HW;
  float* outG = out_chw + 1 * HW;
  float* outB = out_chw + 2 * HW;

  const __m256 meanR = _mm256_set1_ps(mean[0]);
  const __m256 meanG = _mm256_set1_ps(mean[1]);
  const __m256 meanB = _mm256_set1_ps(mean[2]);

  const __m256 invStdR = _mm256_set1_ps(1.0f / stdv[0]);
  const __m256 invStdG = _mm256_set1_ps(1.0f / stdv[1]);
  const __m256 invStdB = _mm256_set1_ps(1.0f / stdv[2]);

  int i = 0;

  // Process 8 pixels at a time.
  // Input layout is interleaved: RGBRGBRGB...
  // We do "gather" loads for R/G/B because memory is not planar.
  // This is not the absolute fastest possible (planar input would be faster),
  // but it is a clean and correct AVX2 improvement.
  for (; i + 8 <= HW; i += 8) {
    // gather indices for r,g,b within 3*HW array
    // offsets in floats:
    // pixel k base = 3*(i+k)
    // r = base + 0, g = base + 1, b = base + 2
    __m256i idx0 = _mm256_setr_epi32(
      3*(i+0)+0, 3*(i+1)+0, 3*(i+2)+0, 3*(i+3)+0,
      3*(i+4)+0, 3*(i+5)+0, 3*(i+6)+0, 3*(i+7)+0
    );
    __m256i idx1 = _mm256_setr_epi32(
      3*(i+0)+1, 3*(i+1)+1, 3*(i+2)+1, 3*(i+3)+1,
      3*(i+4)+1, 3*(i+5)+1, 3*(i+6)+1, 3*(i+7)+1
    );
    __m256i idx2 = _mm256_setr_epi32(
      3*(i+0)+2, 3*(i+1)+2, 3*(i+2)+2, 3*(i+3)+2,
      3*(i+4)+2, 3*(i+5)+2, 3*(i+6)+2, 3*(i+7)+2
    );

    __m256 r = _mm256_i32gather_ps(hwc_rgb, idx0, 4);
    __m256 g = _mm256_i32gather_ps(hwc_rgb, idx1, 4);
    __m256 b = _mm256_i32gather_ps(hwc_rgb, idx2, 4);

    // normalize: (x - mean) * (1/std)
    r = _mm256_mul_ps(_mm256_sub_ps(r, meanR), invStdR);
    g = _mm256_mul_ps(_mm256_sub_ps(g, meanG), invStdG);
    b = _mm256_mul_ps(_mm256_sub_ps(b, meanB), invStdB);

    _mm256_storeu_ps(outR + i, r);
    _mm256_storeu_ps(outG + i, g);
    _mm256_storeu_ps(outB + i, b);
  }

  // remainder
  for (; i < HW; ++i) {
    float r = hwc_rgb[3*i + 0];
    float g = hwc_rgb[3*i + 1];
    float b = hwc_rgb[3*i + 2];

    outR[i] = (r - mean[0]) / stdv[0];
    outG[i] = (g - mean[1]) / stdv[1];
    outB[i] = (b - mean[2]) / stdv[2];
  }
}
#endif

void hwc_rgb_to_chw_normalize(
    const float* hwc_rgb, int H, int W,
    const float mean[3], const float stdv[3],
    float* out_chw
) {
#if defined(__AVX2__)
  hwc_rgb_to_chw_normalize_avx2(hwc_rgb, H, W, mean, stdv, out_chw);
#else
  hwc_rgb_to_chw_normalize_scalar(hwc_rgb, H, W, mean, stdv, out_chw);
#endif
}

} // namespace preproc
