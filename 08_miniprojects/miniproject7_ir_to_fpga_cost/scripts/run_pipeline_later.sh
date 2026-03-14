#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Example: run op counter (requires you to have Relay export output later)
# python ../tvm_relay_intro/scripts/relay_op_counter.py --relay_txt ../tvm_relay_intro/outputs/relay_resnet18_224.txt

# Toy rewrite (example)
# python ../tvm_relay_intro/scripts/relay_pattern_rewrite_toy.py --in_txt ../tvm_relay_intro/outputs/relay_resnet18_224.txt --out_txt ${ROOT}/results/relay_rewritten.txt

# Cost model integration using a demo mapped graph
python "${ROOT}/scripts/relay_to_cost.py" \
  --graph_json "${ROOT}/examples/graph_from_relay_example.json" \
  --out_json "${ROOT}/results/cost.json"
