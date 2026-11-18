# Experiments

Smaller studies and prototypes that back up the main [projects](../projects). Each folder has its own README.
They are less polished than the projects; the **Status** column says honestly what has been run.

**Status key:** ✅ runs on CPU and results are committed · 🔧 bug found and fixed during review · 🟡 needs PyTorch or a GPU, not yet re-run · 📐 design notes / scaffold

## Efficient maths for DNN kernels — [`efficient-math/`](efficient-math)
| Study | Question | Status |
|---|---|---|
| [Winograd F(2×2, 3×3)](efficient-math/winograd) | 3×3 conv with 16 instead of 36 multiplies per 2×2 output tile | ✅ max error 2e-6 vs direct conv |
| [FFT convolution](efficient-math/fft_conv) | At what kernel size does FFT beat direct convolution? | ✅ 🔧 2D version computed true convolution instead of the CNN-style cross-correlation (error 11.4 → 1e-6) |
| [Strassen](efficient-math/strassen) | When does O(N^2.81) beat a tuned BLAS in practice? | ✅ |
| [Low-rank SVD](efficient-math/low_rank_svd) | Parameter/FLOP savings vs accuracy when factorizing linear layers | 🟡 |

## Systems & scheduling — [`systems-scheduling/`](systems-scheduling)
| Study | Question | Status |
|---|---|---|
| [HEFT](systems-scheduling/heft) | List-scheduling a model's task graph across CPU / GPU / FPGA | ✅ example DAG scheduled (makespan 4.8 ms) |
| [Amdahl](systems-scheduling/amdahl), [pipeline model](systems-scheduling/pipeline_model) | How much does accelerating one pipeline stage really help? | ✅ CLI tools |
| [FPGA kernel mapping](systems-scheduling/scheduler_fpga_mapping) | Plug measured/proxy FPGA kernel latencies into the scheduler | 📐 |
| [PCIe vs network-attached accelerators](systems-scheduling/systems) | Transfer-cost models | 📐 |

## Distributed learning — [`distributed-learning/`](distributed-learning)
| Study | Question | Status |
|---|---|---|
| [Ring all-reduce](distributed-learning/distributed) | The NCCL-style collective used in data-parallel training | ✅ 🔧 all-gather used the wrong chunk index, so the simulator failed its own check; fixed and tested for N = 2, 3, 4, 8 |
| [FedAvg](distributed-learning/federated) | Federated averaging, IID vs non-IID clients | ✅ |

## Hardware-aware NAS — [`hardware-aware-nas/`](hardware-aware-nas)
DARTS and latency-aware DARTS, one-shot supernet, REINFORCE controller, zero-cost proxies (SNIP / GradNorm), reduced-training proxies, Pareto filtering, latency-predictor dataset and models, GNN latency-predictor design, HW/SW co-design workflow. Status: 🟡 (PyTorch; synthetic data loaders).

## Model compression — [`model-compression/`](model-compression)
Pruning (unstructured vs structured) and quantization (PTQ vs QAT) on a small model. Status: 🟡

## FPGA cost modelling — [`fpga-cost-modeling/`](fpga-cost-modeling)
Analytical FPGA latency estimators (v0, v1), FPGA-aware NAS ranking, tiling/unroll sweep generator, overlay ISA for Conv/BN/ReLU, async UART host client, small BNN training. Status: 📐 / 🟡

## ML compilers — [`ml-compilers/`](ml-compilers)
TVM Relay introduction, TVM auto-tuning of conv2d, LLVM IR basics. Status: 🟡 (needs TVM)

## Other
- [`benchmarks/`](benchmarks): host↔device transfer benchmark, real-time video classifier demo, performance notebook (🟡)
- [`mini-projects/`](mini-projects): integration exercises combining the studies above (📐)
- [`tinyml-micro/`](tinyml-micro): TFLite Micro C++ deployment notes (📐)
- [`reference/`](reference): INT8 MobileNet-block reference in NumPy

## Reproduce the CPU results
```bash
cd efficient-math/winograd && python src/benchmark_winograd.py     # same pattern for fft_conv, strassen
cd systems-scheduling/heft && python src/run_heft.py
cd distributed-learning/distributed && python scripts/ring_allreduce_sim.py --N 8 --L 4096
```
CPU timings in `results/` were measured on a cloud Intel Xeon @ 2.1 GHz; compare trends, not absolute times.
