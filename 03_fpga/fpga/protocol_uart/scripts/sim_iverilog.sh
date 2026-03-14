#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RTL="${ROOT}/rtl"
TB="${ROOT}/tb"
OUT="${ROOT}/sim_out"
mkdir -p "${OUT}"

iverilog -g2012 -o "${OUT}/sim.out" \
  "${TB}/tb_uart_frame_loopback.sv" \
  "${RTL}/uart_frame_rx.sv" \
  "${RTL}/uart_frame_tx.sv" \
  "${RTL}/crc8.sv"

vvp "${OUT}/sim.out"
