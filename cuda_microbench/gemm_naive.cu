#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

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

static void cpu_gemm(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int M, int N, int K) {
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

    std::printf("Naive GEMM: M=%d N=%d K=%d\n", M, N, K);

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

    gemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    check(cudaGetLastError(), "kernel launch");
    check(cudaDeviceSynchronize(), "sync");

    check(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H C");

    // Reference on CPU (slow, but fine for correctness at these sizes)
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
    std::printf("NOTE: Timing will be added next; today is correctness + baseline implementation.\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
