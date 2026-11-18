// FP32 SGEMM kernels, from simplest to fastest.
//
// All matrices are row-major:  C[M x N] = A[M x K] * B[K x N]
// Every kernel handles arbitrary M, N, K (out-of-range loads are zero-filled).
#pragma once

#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// 1) Naive: one thread computes one element of C, reading A and B straight
//    from global memory. 2*K global loads per 2*K FLOPs -> memory bound.
// -----------------------------------------------------------------------------
__global__ void sgemm_naive(int M, int N, int K, const float* __restrict__ A,
                            const float* __restrict__ B, float* __restrict__ C) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;  // x -> col: coalesced B/C access
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

// -----------------------------------------------------------------------------
// 2) Shared-memory tiling: a block loads TILE x TILE tiles of A and B into
//    shared memory once and reuses each value TILE times.
//    Global traffic drops by a factor of TILE.
// -----------------------------------------------------------------------------
template <int TILE>
__global__ void sgemm_smem_tiled(int M, int N, int K, const float* __restrict__ A,
                                 const float* __restrict__ B, float* __restrict__ C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        const int a_col = t + threadIdx.x;
        const int b_row = t + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// -----------------------------------------------------------------------------
// 3) 2D register blocking: each block computes a BM x BN tile of C and each
//    thread computes a TM x TN micro-tile held in registers.
//    Per inner step a thread loads TM + TN values from shared memory and does
//    TM * TN FMAs, so arithmetic intensity per shared-memory load rises from
//    ~1 to (TM*TN)/(TM+TN). This is the core idea behind cuBLAS-style kernels.
//    Launch with (BM/TM)*(BN/TN) threads in a 1D block.
// -----------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
sgemm_register_blocked(int M, int N, int K, const float* __restrict__ A,
                       const float* __restrict__ B, float* __restrict__ C) {
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
    static_assert(NUM_THREADS % BK == 0, "A-tile loader needs NUM_THREADS % BK == 0");
    static_assert(NUM_THREADS % BN == 0, "B-tile loader needs NUM_THREADS % BN == 0");

    __shared__ float As[BM * BK];  // BM rows x BK cols
    __shared__ float Bs[BK * BN];  // BK rows x BN cols

    const int tid = threadIdx.x;
    const int block_row = blockIdx.y * BM;  // first C row of this block
    const int block_col = blockIdx.x * BN;  // first C col of this block

    // Position of this thread's TM x TN micro-tile inside the block tile.
    const int thread_row = tid / (BN / TN);
    const int thread_col = tid % (BN / TN);

    // Cooperative loading pattern for the shared-memory tiles.
    const int a_load_row = tid / BK, a_load_col = tid % BK;
    constexpr int A_ROW_STRIDE = NUM_THREADS / BK;
    const int b_load_row = tid / BN, b_load_col = tid % BN;
    constexpr int B_ROW_STRIDE = NUM_THREADS / BN;

    float acc[TM][TN] = {};
    float reg_a[TM];
    float reg_b[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load A tile (BM x BK)
        for (int r = a_load_row; r < BM; r += A_ROW_STRIDE) {
            const int g_row = block_row + r, g_col = k0 + a_load_col;
            As[r * BK + a_load_col] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
        }
        // Load B tile (BK x BN)
        for (int r = b_load_row; r < BK; r += B_ROW_STRIDE) {
            const int g_row = k0 + r, g_col = block_col + b_load_col;
            Bs[r * BN + b_load_col] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
#pragma unroll
            for (int i = 0; i < TM; ++i) reg_a[i] = As[(thread_row * TM + i) * BK + kk];
#pragma unroll
            for (int j = 0; j < TN; ++j) reg_b[j] = Bs[kk * BN + thread_col * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += reg_a[i] * reg_b[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int g_row = block_row + thread_row * TM + i;
        if (g_row >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int g_col = block_col + thread_col * TN + j;
            if (g_col < N) C[g_row * N + g_col] = acc[i][j];
        }
    }
}
