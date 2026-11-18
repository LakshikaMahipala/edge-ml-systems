Low-rank SVD Compression 
Includes
- NumPy demo: SVD energy and approximation error
- PyTorch utility: compress nn.Linear into two smaller Linear layers
- Benchmark harness: timing + error across ranks

Run later
python src/benchmark_low_rank.py --in_dim 1024 --out_dim 1024 --batch 32 --ranks 64,128,256,512
