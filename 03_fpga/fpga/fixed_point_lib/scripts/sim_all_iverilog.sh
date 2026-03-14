#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RTL="${ROOT}/rtl"
TB="${ROOT}/tb"
OUT="${ROOT}/sim_out"
mkdir -p "${OUT}"

common="${RTL}/fxp_pkg.sv ${RTL}/fxp_saturate.sv ${RTL}/fxp_add.sv ${RTL}/fxp_mul.sv ${RTL}/fxp_scale_shift.sv"

run_one () {
  name="$1"
  tbfile="$2"
  echo "== $name =="
  iverilog -g2012 -o "${OUT}/${name}.out" ${tbfile} ${common}
  vvp "${OUT}/${name}.out"
  echo ""
}

run_one tb_fxp_add       "${TB}/tb_fxp_add.sv"
run_one tb_fxp_mul       "${TB}/tb_fxp_mul.sv"
run_one tb_fxp_saturate  "${TB}/tb_fxp_saturate.sv"
run_one tb_fxp_scale     "${TB}/tb_fxp_scale_shift.sv"

echo "ALL FIXED-POINT TESTS PASS"
