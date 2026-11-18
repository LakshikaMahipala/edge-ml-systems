Winograd Convolution 

What this contains
- Winograd F(2x2,3x3) reference implementation (single-channel)
- Naive conv reference
- Benchmark harness (run later)
- Docs: intuition, transforms, precision regimes, FPGA implications

Run later
python src/benchmark_winograd.py --H 34 --W 34 --out_csv results/winograd_results.csv
