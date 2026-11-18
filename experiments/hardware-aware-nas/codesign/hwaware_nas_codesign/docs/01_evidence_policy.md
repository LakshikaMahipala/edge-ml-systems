# Evidence policy (critical)

This repo contains:
- real code implementations
- proxy-based evaluations (until local/FPGA measurement is available)

Rules:
1) Any latency/energy values are labeled:
   - "proxy" unless measured on hardware
2) Speedup claims are forbidden without:
   - device, precision, clock, IO regime, batch size
3) NAS rankings are reported with:
   - ranking metrics (Spearman, Top-K overlap)
4) Hardware-aware claims require later validation:
   - measured kernel points and/or synthesis reports
