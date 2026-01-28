#!/usr/bin/env bash
set -euo pipefail

# This script expects you built binaries in: cuda_microbench/build
# Run later after local CUDA setup.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
OUT_DIR="${ROOT_DIR}/results"

mkdir -p "${OUT_DIR}"

ts="$(date +%Y%m%d_%H%M%S)"
out="${OUT_DIR}/run_${ts}.txt"

echo "CUDA Microbench Suite Run @ ${ts}" | tee "${out}"
echo "Build dir: ${BUILD_DIR}" | tee -a "${out}"
echo "" | tee -a "${out}"

# Standard sizes (you can change later)
M=1024
N=1024
K=1024
WARMUP=10
ITERS=50

echo "== vector_add ==" | tee -a "${out}"
"${BUILD_DIR}/vector_add" $((1<<20)) | tee -a "${out}"
echo "" | tee -a "${out}"

echo "== gemm_naive M=N=K=${M} ==" | tee -a "${out}"
"${BUILD_DIR}/gemm_naive" "${M}" "${N}" "${K}" "${WARMUP}" "${ITERS}" | tee -a "${out}"
echo "" | tee -a "${out}"

echo "== gemm_tiled M=N=K=${M} ==" | tee -a "${out}"
"${BUILD_DIR}/gemm_tiled" "${M}" "${N}" "${K}" "${WARMUP}" "${ITERS}" | tee -a "${out}"
echo "" | tee -a "${out}"

echo "== gemm_cublas M=N=K=${M} ==" | tee -a "${out}"
"${BUILD_DIR}/gemm_cublas" "${M}" "${N}" "${K}" "${WARMUP}" "${ITERS}" | tee -a "${out}"
echo "" | tee -a "${out}"

echo "== conv_lite_gemm (C=3,H=W=32,F=8,KH=KW=3) ==" | tee -a "${out}"
"${BUILD_DIR}/conv_lite_gemm" 3 32 32 8 3 3 "${WARMUP}" "${ITERS}" | tee -a "${out}"
echo "" | tee -a "${out}"

echo "DONE. Output: ${out}"
echo "Next: parse to CSV -> plot"
