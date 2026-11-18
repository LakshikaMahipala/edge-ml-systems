# Unified results schema

We unify all experiments into one CSV so later we can:
- plot comparisons
- train meta predictors
- publish a clean portfolio table

Columns
- technique: {strassen, winograd, fft_conv, low_rank_svd}
- experiment_id: string (e.g., "N=512_leaf=64", "H=34_W=34", "N=4096_K=127", "in=1024_out=1024_r=128")
- method: method name (e.g., numpy_dot, strassen, naive, winograd, fft_pow2, baseline, low_rank)
- p50_ms, p99_ms, mean_ms
- error: numeric error (max_abs_error or rel_err)
- notes: free text
