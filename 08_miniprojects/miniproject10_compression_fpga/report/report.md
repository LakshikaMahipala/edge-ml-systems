# Mini-project 10: Compression → FPGA evidence 

## 1. Overview
This mini-project consolidates quantization, pruning, and binary neural networks (BNN),
and connects them to FPGA implementation evidence via an RTL XNOR-popcount kernel.

## 2. Quantization (PTQ vs QAT)
- PTQ: post-training calibration, faster pipeline, can lose accuracy
- QAT: trains with fake-quant noise, preserves INT8 accuracy better

Repo artifacts:
- compression_quantization/docs/*
- compression_quantization/qat_small_model/*

Planned results:
- FP32 vs QAT-INT8 accuracy
- timing proxy (CPU) and later backend timing

## 3. Pruning (unstructured vs structured)
Key point:
- Unstructured sparsity compresses but often doesn't speed up without sparse kernels.
- Structured pruning changes shapes and can speed up dense compute on hardware.

Repo artifacts:
- compression_pruning/docs/*
- compression_pruning/pruning_small_model/*

Planned results:
- accuracy vs sparsity/keep_ratio
- timing change (structured expected to help more)

## 4. BNN (binary weights/activations)
Key point:
- BNN replaces multiply with XNOR-popcount.
- Training requires STE, often needs BN and wider layers.

Repo artifacts:
- bnn/docs/*
- bnn/bnn_small_model/*

Planned results:
- FP32 vs BNN accuracy and proxy timing

## 5. LUTNet direction
Key point:
- FPGA LUTs can implement small-k Boolean functions directly.
- High fan-in prevents mapping full neurons directly; need decomposition and training tricks.

Repo artifacts:
- lutnet_notes/docs/*

## 6. FPGA evidence: XNOR-popcount dot kernel
Implemented:
- Synthesizable RTL + testbench
- Python golden reference for exact correctness

Repo artifacts:
- fpga_bnn_xnor_popcount/*

Planned results:
- simulation pass/fail
- cycle count and throughput estimates later

## 7. Evidence rules (honesty)
We do not claim speedups without measurement context.
See: docs/02_hw_claims_and_evidence_rules.md

## 8. Run-later checklist
See: docs/01_results_plan_run_later.md
