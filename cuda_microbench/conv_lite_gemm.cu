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

// X: (C,H,W) row-major: x[c][h][w] -> X[c*H*W + h*W + w]
// im2col output A: (Kdim, OH*OW) where Kdim = C*KH*KW
static void im2col_cpu(const std::vector<float>& X, int C, int H, int W,
                       int KH, int KW,
                       std::vector<float>& A, int OH, int OW) {
    int Kdim = C * KH * KW;
    int cols = OH * OW;
    A.assign(Kdim * cols, 0.0f);

    for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
            int col_idx = oh * OW + ow; // 0..(OH*OW-1)

            int kbase = 0;
            for (int c = 0; c < C; ++c) {
                for (int kh = 0; kh < KH; ++kh) {
                    for (int kw = 0; kw < KW; ++kw) {
                        int ih = oh + kh;
                        int iw = ow + kw;
                        float v = X[c * H * W + ih * W + iw];
                        A[(kbase) * cols + col_idx] = v; // (k, col)
                        kbase++;
                    }
                }
            }
        }
    }
}

// W: (F,C,KH,KW) -> B: (F, Kdim)
static void filters_to_matrix(const std::vector<float>& Wf, int F, int C, int KH, int KW,
                              std::vector<float>& B) {
    int Kdim = C * KH * KW;
    B.assign(F * Kdim, 0.0f);
    for (int f = 0; f < F; ++f) {
        for (int c = 0; c < C; ++c) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    int k = ((c * KH + kh) * KW + kw);
                    B[f * Kdim + k] = Wf[(((f * C + c) * KH + kh) * KW + kw)];
                }
            }
        }
    }
}

// CPU reference conv (valid conv, stride=1, pad=0): Y (F,OH,OW)
static void conv2d_cpu(const std::vector<float>& X, const std::vector<float>& Wf,
                       int C, int H, int W, int F, int KH, int KW,
                       std::vector<float>& Y, int OH, int OW) {
    Y.assign(F * OH * OW, 0.0f);

    for (int f = 0; f < F; ++f) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                float acc = 0.0f;
                for (int c = 0; c < C; ++c) {
                    for (int kh = 0; kh < KH; ++kh) {
                        for (int kw = 0; kw < KW; ++kw) {
                            int ih = oh + kh;
                            int iw = ow + kw;
                            float xv = X[c * H * W + ih * W + iw];
                            float wv = Wf[(((f * C + c) * KH + kh) * KW + kw)];
                            acc += xv * wv;
                        }
                    }
                }
                Y[(f * OH + oh) * OW + ow] = acc;
            }
        }
    }
}

int main(int argc, char** argv) {
    // Default small conv
    int C  = (argc > 1) ? std::atoi(argv[1]) : 3;
    int H  = (argc > 2) ? std::atoi(argv[2]) : 32;
    int W  = (argc > 3) ? std::atoi(argv[3]) : 32;
    int F  = (argc > 4) ? std::atoi(argv[4]) : 8;
    int KH = (argc > 5) ? std::atoi(argv[5]) : 3;
    int KW = (argc > 6) ? std::atoi(argv[6]) : 3;

    int OH = H - KH + 1;
    int OW = W - KW + 1;
    if (OH <= 0 || OW <= 0) {
        std::fprintf(stderr, "Invalid shapes: H,W must be >= KH,KW\n");
        return 2;
    }

    int Kdim = C * KH * KW;
    int cols = OH * OW;

    std::printf("Conv-lite GEMM: C=%d H=%d W=%d F=%d KH=%d KW=%d => OH=%d OW=%d\n", C,H,W,F,KH,KW,OH,OW);
    std::printf("GEMM shapes: B(F x Kdim) * A(Kdim x cols) = Y(F x cols)\n");
    std::printf("Kdim=%d cols=%d\n", Kdim, cols);

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> X(C * H * W);
    std::vector<float> Wf(F * C * KH * KW);
    for (auto& x : X)  x  = dist(rng);
    for (auto& w : Wf) w = dist(rng);

    // CPU reference conv
    std::vector<float> Yref;
    conv2d_cpu(X, Wf, C, H, W, F, KH, KW, Yref, OH, OW);

    // im2col + filter reshape
    std::vector<float> A; // (Kdim, cols)
    std::vector<float> B; // (F, Kdim)
    im2col_cpu(X, C, H, W, KH, KW, A, OH, OW);
    filters_to_matrix(Wf, F, C, KH, KW, B);

    // Device buffers
    float *dA=nullptr, *dB=nullptr, *dY=nullptr;
    check(cudaMalloc(&dA, A.size() * sizeof(float)), "cudaMalloc dA");
    check(cudaMalloc(&dB, B.size() * sizeof(float)), "cudaMalloc dB");
    check(cudaMalloc(&dY, (F * cols) * sizeof(float)), "cudaMalloc dY");

    check(cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D A");
    check(cudaMemcpy(dB, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice), "H2D B");

    // cuBLAS GEMM: Y = B * A
    // Our arrays are row-major, but cuBLAS assumes column-major.
    // Use transpose trick: (Y^T) = (A^T) * (B^T)
    // Shapes:
    // A: (Kdim x cols) row-major -> A^T: (cols x Kdim) col-major view
    // B: (F x Kdim) row-major -> B^T: (Kdim x F) col-major view
    // Y^T: (cols x F)

    cublasHandle_t handle;
    check_cublas(cublasCreate(&handle), "cublasCreate");

    float alpha = 1.0f;
    float beta  = 0.0f;

    // Compute Y^T (cols x F) = A^T (cols x Kdim) * B^T (Kdim x F)
    // cublasSgemm column-major:
    // C(m x n) = A(m x k) * B(k x n)
    // Here: m=cols, n=F, k=Kdim
    check_cublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            cols, F, Kdim,
            &alpha,
            dA, cols,   // A^T leading dimension = cols
            dB, Kdim,   // B^T leading dimension = Kdim
            &beta,
            dY, cols    // Y^T leading dimension = cols
        ),
        "cublasSgemm conv-lite"
    );

    check(cudaDeviceSynchronize(), "sync");

    // Copy Y back (stored as row-major (F x cols) in host view after transpose trick)
    std::vector<float> Y(F * cols, 0.0f);
    check(cudaMemcpy(Y.data(), dY, Y.size() * sizeof(float), cudaMemcpyDeviceToHost), "D2H Y");

    // Compare Y (F x cols) with Yref (F x OH x OW)
    double max_abs_err = 0.0, max_rel_err = 0.0;
    for (int f = 0; f < F; ++f) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                int col = oh * OW + ow;
                double a = Yref[(f * OH + oh) * OW + ow];
                double b = Y[f * cols + col];
                double abs_err = std::fabs(a - b);
                double rel_err = abs_err / (std::fabs(a) + 1e-6);
                max_abs_err = std::max(max_abs_err, abs_err);
                max_rel_err = std::max(max_rel_err, rel_err);
            }
        }
    }

    std::printf("Correctness: max_abs_err=%.6e max_rel_err=%.6e\n", max_abs_err, max_rel_err);
    std::printf("NOTE: Timing will be collected on Nsight day and microbench suite day.\n");

    cublasDestroy(handle);
    cudaFree(dA); cudaFree(dB); cudaFree(dY);
    return 0;
}
