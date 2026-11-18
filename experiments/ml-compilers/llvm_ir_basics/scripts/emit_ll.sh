#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/outputs"
mkdir -p "${OUT}"

clang -S -emit-llvm -O0 "${ROOT}/src/vec_add.c" -o "${OUT}/vec_add_O0.ll"
clang -S -emit-llvm -O3 "${ROOT}/src/vec_add.c" -o "${OUT}/vec_add_O3.ll"

clang -S -emit-llvm -O0 "${ROOT}/src/matmul_naive.c" -o "${OUT}/matmul_naive_O0.ll"
clang -S -emit-llvm -O3 "${ROOT}/src/matmul_naive.c" -o "${OUT}/matmul_naive_O3.ll"

echo "Wrote LLVM IR into ${OUT}"
