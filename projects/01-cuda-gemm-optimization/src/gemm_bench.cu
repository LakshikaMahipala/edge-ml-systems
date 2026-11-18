// Benchmarks every SGEMM kernel against cuBLAS over a sweep of square sizes.
// Each kernel's output is checked against cuBLAS before its time is reported.
//
// Usage:  ./gemm_bench [sizes=256,512,1024,2048,4096] [warmup=5] [iters=20]
// Output: CSV on stdout:
//   kernel,M,N,K,time_ms,gflops,pct_of_cublas,max_norm_err,correct

#include "common.cuh"
#include "gemm_kernels.cuh"

#include <sstream>
#include <string>

namespace {

constexpr double kTolerance = 1e-4;  // normalised error vs cuBLAS (FP32)

std::vector<int> parse_sizes(const char* arg) {
    std::vector<int> sizes;
    std::stringstream ss(arg);
    std::string item;
    while (std::getline(ss, item, ',')) sizes.push_back(std::stoi(item));
    return sizes;
}

// Row-major C = A*B with column-major cuBLAS: compute C^T = B^T * A^T.
void cublas_sgemm(cublasHandle_t h, int M, int N, int K, const float* A,
                  const float* B, float* C) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N,
                             A, K, &beta, C, N));
}

int ceil_div(int a, int b) { return (a + b - 1) / b; }

}  // namespace

int main(int argc, char** argv) {
    const std::vector<int> sizes =
        parse_sizes(argc > 1 ? argv[1] : "256,512,1024,2048,4096");
    const int warmup = argc > 2 ? std::atoi(argv[2]) : 5;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 20;

    print_device_info();
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    std::printf("kernel,M,N,K,time_ms,gflops,pct_of_cublas,max_norm_err,correct\n");
    bool all_correct = true;

    for (int n : sizes) {
        const int M = n, N = n, K = n;
        std::vector<float> hA(size_t(M) * K), hB(size_t(K) * N);
        std::vector<float> hRef(size_t(M) * N), hC(size_t(M) * N);
        fill_uniform(hA, 1);
        fill_uniform(hB, 2);

        float *dA, *dB, *dC, *dRef;
        CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dRef, hRef.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

        const double flops = 2.0 * M * N * double(K);

        // Reference: cuBLAS
        const float cublas_ms =
            time_ms([&] { cublas_sgemm(handle, M, N, K, dA, dB, dRef); }, warmup, iters);
        CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, hRef.size() * sizeof(float), cudaMemcpyDeviceToHost));

        auto report = [&](const char* name, float ms, bool check) {
            double err = 0.0;
            bool ok = true;
            if (check) {
                CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));
                err = normalized_max_error(hC, hRef);
                ok = err < kTolerance;
                all_correct = all_correct && ok;
            }
            const double gflops = flops / (ms * 1e6);
            std::printf("%s,%d,%d,%d,%.4f,%.1f,%.1f,%.2e,%s\n", name, M, N, K, ms,
                        gflops, 100.0 * cublas_ms / ms, err, ok ? "yes" : "NO");
            std::fflush(stdout);
        };

        // 1) Naive
        {
            dim3 block(32, 8);
            dim3 grid(ceil_div(N, block.x), ceil_div(M, block.y));
            CUDA_CHECK(cudaMemset(dC, 0, hC.size() * sizeof(float)));
            float ms = time_ms([&] { sgemm_naive<<<grid, block>>>(M, N, K, dA, dB, dC); },
                               warmup, iters);
            report("naive", ms, true);
        }
        // 2) Shared-memory tiling
        {
            constexpr int TILE = 32;
            dim3 block(TILE, TILE);
            dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
            CUDA_CHECK(cudaMemset(dC, 0, hC.size() * sizeof(float)));
            float ms = time_ms(
                [&] { sgemm_smem_tiled<TILE><<<grid, block>>>(M, N, K, dA, dB, dC); },
                warmup, iters);
            report("smem_tiled_32", ms, true);
        }
        // 3) 2D register blocking
        {
            constexpr int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
            dim3 block((BM / TM) * (BN / TN));
            dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
            CUDA_CHECK(cudaMemset(dC, 0, hC.size() * sizeof(float)));
            float ms = time_ms(
                [&] {
                    sgemm_register_blocked<BM, BN, BK, TM, TN>
                        <<<grid, block>>>(M, N, K, dA, dB, dC);
                },
                warmup, iters);
            report("register_blocked_64x64", ms, true);
        }
        report("cublas", cublas_ms, false);

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
        CUDA_CHECK(cudaFree(dRef));
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    if (!all_correct) {
        std::fprintf(stderr, "# ERROR: at least one kernel failed the correctness check\n");
        return 1;
    }
    std::fprintf(stderr, "# All kernels match cuBLAS (normalised error < %.0e)\n", kTolerance);
    return 0;
}
