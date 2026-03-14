#include <stddef.h>

void matmul_naive(const float* A, const float* B, float* C,
                  size_t M, size_t N, size_t K) {
    // C[M,N] = A[M,K] * B[K,N], row-major
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float acc = 0.0f;
            for (size_t k = 0; k < K; k++) {
                acc += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = acc;
        }
    }
}
