#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/outputs"
mkdir -p "${OUT}"

# Emit bitcode, run opt passes, then disassemble
clang -c -emit-llvm -O0 "${ROOT}/src/vec_add.c" -o "${OUT}/vec_add_O0.bc"
opt -O3 "${OUT}/vec_add_O0.bc" -o "${OUT}/vec_add_opt.bc"
llvm-dis "${OUT}/vec_add_opt.bc" -o "${OUT}/vec_add_opt.ll"

echo "Wrote optimized IR into ${OUT}/vec_add_opt.ll"
