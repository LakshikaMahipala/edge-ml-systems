FFT Convolution 

Includes
- 1D full convolution: naive vs FFT (rfft/irfft)
- 2D single-channel valid convolution: naive vs FFT crop
- Benchmark harness to observe crossover regimes

Run later
python src/benchmark_fft_conv.py --out_csv results/fft_conv_1d_crossover.csv
