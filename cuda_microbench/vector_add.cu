#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>

__global__ void vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

static void check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);

    // Host buffers
    std::vector<float> ha(n, 1.0f), hb(n, 2.0f), hc(n, 0.0f);

    // Device buffers
    float *da=nullptr, *db=nullptr, *dc=nullptr;
    check(cudaMalloc(&da, n * sizeof(float)), "cudaMalloc da");
    check(cudaMalloc(&db, n * sizeof(float)), "cudaMalloc db");
    check(cudaMalloc(&dc, n * sizeof(float)), "cudaMalloc dc");

    // H2D
    check(cudaMemcpy(da, ha.data(), n * sizeof(float), cudaMemcpyHostToDevice), "H2D a");
    check(cudaMemcpy(db, hb.data(), n * sizeof(float), cudaMemcpyHostToDevice), "H2D b");

    // Launch
    int block = 256;
    int grid = (n + block - 1) / block;

    vadd<<<grid, block>>>(da, db, dc, n);
    check(cudaGetLastError(), "kernel launch");
    check(cudaDeviceSynchronize(), "sync");

    // D2H
    check(cudaMemcpy(hc.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost), "D2H c");

    // Verify first few
    for (int i = 0; i < 10; ++i) {
        if (std::fabs(hc[i] - 3.0f) > 1e-6f) {
            std::fprintf(stderr, "Mismatch at %d: %f\n", i, hc[i]);
            return 2;
        }
    }

    std::printf("OK. n=%d, c[0]=%f\n", n, hc[0]);

    cudaFree(da); cudaFree(db); cudaFree(dc);
    return 0;
}
