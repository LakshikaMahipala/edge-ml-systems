#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RTL="${ROOT}/rtl"
TB="${ROOT}/tb"

OUT="${ROOT}/sim_out"
mkdir -p "${OUT}"

iverilog -g2012 -o "${OUT}/sim.out" \
  "${TB}/tb_int8_fc.sv" \
  "${RTL}/int8_dot.sv" \
  "${RTL}/int8_requant.sv" \
  "${RTL}/int8_fc.sv"

vvp "${OUT}/sim.out"
