	cuda_vector_add | n=1<<20 block=256 | time_ms=TBD 
	gemm_naive | M=N=K=512 | correctness=OK | time_ms=TBD | gflops=TBD 
	gemm_cublas | M=N=K=512 | correctness=OK | time_ms=TBD | gflops=TBD 
	speedup | cublas/naive | time_ms ratio TBD 
	gemm_tiled | 512^3 | time_ms=TBD | gflops=TBD | speedup_vs_naive=TBD 
	conv_lite_gemm | C=3 H=W=32 F=8 KH=KW=3 | correctness=OK | time_ms=TBD 

