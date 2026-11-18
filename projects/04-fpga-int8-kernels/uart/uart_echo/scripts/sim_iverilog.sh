#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RTL="${ROOT}/rtl"
TB="${ROOT}/tb"

OUT="${ROOT}/sim_out"
mkdir -p "${OUT}"

echo "[SIM] Building..."
iverilog -g2012 -o "${OUT}/sim" \
  "${TB}/tb_uart_echo.sv" \
  "${RTL}/uart_rx.v" \
  "${RTL}/uart_tx.v" \
  "${RTL}/uart_echo_top.v"

echo "[SIM] Running..."
vvp "${OUT}/sim"
echo "[SIM] DONE"
