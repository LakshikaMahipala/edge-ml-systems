#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

#include "src/cuda_events.h"

static void check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

__global__ void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    // C[MxN] = A[MxK] * B[KxN] (row-major arrays)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // [0..M)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // [0..N)

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

static void cpu_gemm(const std::vector<float>& A, const std::vector<float>& B,
                     std::vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? std::atoi(argv[1]) : 512;
    int N = (argc > 2) ? std::atoi(argv[2]) : 512;
    int K = (argc > 3) ? std::atoi(argv[3]) : 512;

    int warmup = (argc > 4) ? std::atoi(argv[4]) : 10;
    int iters  = (argc > 5) ? std::atoi(argv[5]) : 50;

    std::printf("Naive GEMM: M=%d N=%d K=%d | warmup=%d iters=%d\n", M, N, K, warmup, iters);

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> hA(M * K), hB(K * N), hC(M * N, 0.0f), hCref(M * N, 0.0f);
    for (auto& x : hA) x = dist(rng);
    for (auto& x : hB) x = dist(rng);

    float *dA=nullptr, *dB=nullptr, *dC=nullptr;
    check(cudaMalloc(&dA, hA.size() * sizeof(float)), "cudaMalloc dA");
    check(cudaMalloc(&dB, hB.size() * sizeof(float)), "cudaMalloc dB");
    check(cudaMalloc(&dC, hC.size() * sizeof(float)), "cudaMalloc dC");

    check(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D A");
    check(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D B");

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // Warmup (not timed)
    for (int i = 0; i < warmup; ++i) {
        gemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    check(cudaGetLastError(), "warmup kernel launch");
    check(cudaDeviceSynchronize(), "warmup sync");

    // Timed loop (kernel-only)
    CudaEventTimer t;
    t.tic();
    for (int i = 0; i < iters; ++i) {
        gemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    check(cudaGetLastError(), "timed kernel launch");
    float ms_total = t.toc_ms();
    check(cudaDeviceSynchronize(), "final sync");

    float avg_ms = ms_total / iters;
    double flops = 2.0 * double(M) * double(N) * double(K);
    double gflops = (flops / 1e9) / (avg_ms / 1e3);
    std::printf("Kernel avg time: %.3f ms | Throughput: %.2f GFLOP/s\n", avg_ms, gflops);

    // Copy result once for correctness
    check(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H C");

    // CPU reference (slow but fine at moderate sizes)
    cpu_gemm(hA, hB, hCref, M, N, K);

    // Compare
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    for (int i = 0; i < M * N; ++i) {
        double a = hCref[i];
        double b = hC[i];
        double abs_err = std::fabs(a - b);
        double rel_err = abs_err / (std::fabs(a) + 1e-6);
        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
    }

    std::printf("Correctness: max_abs_err=%.6e max_rel_err=%.6e\n", max_abs_err, max_rel_err);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
