# Run-later results plan 

## Quantization (QAT)
Run:
- compression_quantization/qat_small_model/src/train_fp32.py
- compression_quantization/qat_small_model/src/train_qat.py

Record:
- val_acc FP32 vs INT8(QAT)
- p50 latency proxy (CPU)

## Pruning
Run:
- compression_pruning/pruning_small_model/src/train_base.py
- prune_unstructured.py (various sparsities)
- prune_structured.py (various keep ratios)

Record:
- accuracy drop vs sparsity
- timing change (expect: structured helps more)

## BNN
Run:
- bnn/bnn_small_model/src/train_fp32.py
- train_bnn.py

Record:
- accuracy gap FP32 vs BNN
- timing proxy (CPU; note: true speedup requires FPGA)

## FPGA BNN kernel
Sim:
- fpga_bnn_xnor_popcount/sim/tb_bnn_dot_top.v

Record:
- pass/fail correctness
- later: cycles per dot product and estimated throughput at f_clk
