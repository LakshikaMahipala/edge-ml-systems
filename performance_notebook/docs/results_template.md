Results Template (fill after local runs)

Table A — Stage breakdown (p50 ms)

| Backend | pre | infer | post | io/copy | e2e |
|--------|-----:|------:|-----:|--------:|----:|
| PyTorch CPU | TBD | TBD | TBD | TBD | TBD |
| ORT CPU | TBD | TBD | TBD | TBD | TBD |
| TRT FP32 | TBD | TBD | TBD | TBD | TBD |
| TRT FP16 | TBD | TBD | TBD | TBD | TBD |
| TRT INT8 | TBD | TBD | TBD | TBD | TBD |
| FPGA UART v0/v1 | TBD | TBD | TBD | TBD | TBD |

Table B — Tail latency (p99 ms)

| Backend | pre p99 | infer p99 | post p99 | io/copy p99 | e2e p99 |
|--------|--------:|----------:|---------:|------------:|--------:|
| PyTorch CPU | TBD | TBD | TBD | TBD | TBD |
| ... | ... | ... | ... | ... | ... |

Table C — Amdahl bounds (inference acceleration)

| Backend | p = infer/total | max speedup (inf→∞) | measured/assumed S_inf | predicted speedup |
|--------|-----------------:|---------------------:|------------------------:|------------------:|
| baseline | TBD | TBD | 1.0 | 1.0 |
| TRT FP16 | TBD | TBD | TBD | TBD |
| TRT INT8 | TBD | TBD | TBD | TBD |
| FPGA | TBD | TBD | TBD | TBD |
