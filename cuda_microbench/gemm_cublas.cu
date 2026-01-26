#include <cuda_runtime.h>
#include <cublas_v2.h>
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

static void check_cublas(cublasStatus_t s, const char* msg) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error: %s (status=%d)\n", msg, int(s));
        std::exit(1);
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

    std::printf("cuBLAS GEMM: M=%d N=%d K=%d\n", M, N, K);

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // We keep host data in row-major for reference.
    std::vector<float> hA(M * K), hB(K * N), hC(M * N, 0.0f), hCref(M * N, 0.0f);
    for (auto& x : hA) x = dist(rng);
    for (auto& x : hB) x = dist(rng);

    float *dA=nullptr, *dB=nullptr, *dC=nullptr;
    check(cudaMalloc(&dA, hA.size() * sizeof(float)), "cudaMalloc dA");
    check(cudaMalloc(&dB, hB.size() * sizeof(float)), "cudaMalloc dB");
    check(cudaMalloc(&dC, hC.size() * sizeof(float)), "cudaMalloc dC");

    check(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D A");
    check(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D B");

    // cuBLAS assumes column-major by default.
    // Trick: compute C_rowmajor = A_rowmajor * B_rowmajor
    // by interpreting row-major matrices as transposed column-major.
    // We can use:
    //   C^T = B^T * A^T
    // and call cublasSgemm with swapped A/B.
    //
    // Using column-major:
    //   (N x M) = (N x K) * (K x M)
    // where:
    //   B is (K x N) row-major -> B^T is (N x K) column-major view
    //   A is (M x K) row-major -> A^T is (K x M) column-major view

    cublasHandle_t handle;
    check_cublas(cublasCreate(&handle), "cublasCreate");

    float alpha = 1.0f;
    float beta  = 0.0f;

    // Compute C^T (N x M) into dC but stored as row-major (M x N) on host view after copy.
    check_cublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            dB, N,   // B^T as (N x K) in column-major => leading dim N
            dA, K,   // A^T as (K x M) in column-major => leading dim K
            &beta,
            dC, N    // C^T as (N x M), leading dim N
        ),
        "cublasSgemm"
    );

    check(cudaDeviceSynchronize(), "sync");
    check(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H C");

    cpu_gemm(hA, hB, hCref, M, N, K);

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
    std::printf("NOTE: Timing will be added next; today is correctness + baseline cuBLAS.\n");

    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
