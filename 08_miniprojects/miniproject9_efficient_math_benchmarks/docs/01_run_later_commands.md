# Run-later commands (Mini-project 9)

## Strassen
python efficient_math/strassen/src/benchmark_strassen.py --out_csv efficient_math/strassen/results/strassen_results.csv

## Winograd
python efficient_math/winograd/src/benchmark_winograd.py --H 34 --W 34 --out_csv efficient_math/winograd/results/winograd_results.csv

## FFT convolution
python efficient_math/fft_conv/src/benchmark_fft_conv.py --out_csv efficient_math/fft_conv/results/fft_conv_1d_crossover.csv

## Low-rank SVD
python efficient_math/low_rank_svd/src/benchmark_low_rank.py --out_csv efficient_math/low_rank_svd/results/low_rank_results.csv

## Consolidate into unified schema
python miniproject9_efficient_math_benchmarks/src/unify_schema.py \
  --strassen efficient_math/strassen/results/strassen_results.csv \
  --winograd efficient_math/winograd/results/winograd_results.csv \
  --fft efficient_math/fft_conv/results/fft_conv_1d_crossover.csv \
  --svd efficient_math/low_rank_svd/results/low_rank_results.csv \
  --out miniproject9_efficient_math_benchmarks/results/unified_results.csv
