// Shared helpers: error checking, CUDA-event timing, random init, error metrics.
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call,         \
                         __FILE__, __LINE__, cudaGetErrorString(err_));         \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st_ = (call);                                            \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "cuBLAS error %s at %s:%d: status %d\n",       \
                         #call, __FILE__, __LINE__, static_cast<int>(st_));     \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// Times a callable with CUDA events: `warmup` untimed runs, then the mean of
// `iters` timed runs, in milliseconds. Measures GPU time only (no host copies).
template <typename F>
float time_ms(F&& launch, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / static_cast<float>(iters);
}

inline void fill_uniform(std::vector<float>& v, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

// Max absolute difference normalised by the largest reference magnitude.
// Robust to the different floating-point summation orders of each kernel.
inline double normalized_max_error(const std::vector<float>& out,
                                   const std::vector<float>& ref) {
    double max_diff = 0.0, max_ref = 1e-12;
    for (size_t i = 0; i < ref.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(double(out[i]) - double(ref[i])));
        max_ref = std::max(max_ref, std::fabs(double(ref[i])));
    }
    return max_diff / max_ref;
}

inline void print_device_info() {
    int dev = 0;
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    std::fprintf(stderr, "# GPU: %s | SM %d.%d | %d SMs | %.1f GB\n", p.name,
                 p.major, p.minor, p.multiProcessorCount,
                 p.totalGlobalMem / 1073741824.0);
}
