#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RTL="${ROOT}/rtl"
TB="${ROOT}/tb"
OUT="${ROOT}/sim_out"
mkdir -p "${OUT}"

iverilog -g2012 -o "${OUT}/sim.out" \
  "${TB}/tb_dwconv1d_int8.sv" \
  "${RTL}/dwconv1d_int8.sv"

vvp "${OUT}/sim.out"
