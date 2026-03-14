#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

static inline void cuda_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct CudaEventTimer {
    cudaEvent_t start{};
    cudaEvent_t stop{};

    CudaEventTimer() {
        cuda_check(cudaEventCreate(&start), "cudaEventCreate start");
        cuda_check(cudaEventCreate(&stop), "cudaEventCreate stop");
    }

    ~CudaEventTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void tic(cudaStream_t stream = 0) {
        cuda_check(cudaEventRecord(start, stream), "cudaEventRecord start");
    }

    float toc_ms(cudaStream_t stream = 0) {
        cuda_check(cudaEventRecord(stop, stream), "cudaEventRecord stop");
        cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
        float ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
        return ms;
    }
};
