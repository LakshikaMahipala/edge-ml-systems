// Memory-bandwidth baseline: c = a + b.
// Vector add does 1 FLOP per 12 bytes moved, so its speed is set entirely by
// DRAM bandwidth. The achieved GB/s is the practical bandwidth ceiling used in
// the roofline analysis of the GEMM kernels.
//
// Usage: ./vector_add [n_elements=67108864] [warmup=5] [iters=50]

#include "common.cuh"

__global__ void vector_add(const float* __restrict__ a, const float* __restrict__ b,
                           float* __restrict__ c, int n) {
    // Grid-stride loop: any grid size covers any n.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        c[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : (1 << 26);
    const int warmup = argc > 2 ? std::atoi(argv[2]) : 5;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 50;
    print_device_info();

    std::vector<float> ha(n), hb(n), hc(n);
    fill_uniform(ha, 1);
    fill_uniform(hb, 2);

    float *da, *db, *dc;
    const size_t bytes = size_t(n) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));
    CUDA_CHECK(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));

    int sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const int block = 256, grid = sms * 32;

    const float ms =
        time_ms([&] { vector_add<<<grid, block>>>(da, db, dc, n); }, warmup, iters);
    CUDA_CHECK(cudaMemcpy(hc.data(), dc, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < n; ++i)
        if (hc[i] != ha[i] + hb[i]) { ok = false; break; }

    const double gbps = 3.0 * bytes / (ms * 1e6);  // read a, read b, write c
    std::printf("kernel,n,time_ms,bandwidth_GBps,correct\n");
    std::printf("vector_add,%d,%.4f,%.1f,%s\n", n, ms, gbps, ok ? "yes" : "NO");

    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    return ok ? 0 : 1;
}
