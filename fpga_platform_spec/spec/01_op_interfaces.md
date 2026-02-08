FPGA Operator Interfaces (v0)

General conventions
- int8 activations
- int8 weights
- int32 accumulation
- output int8 via SHIFT requant + saturate
- batch=1
- static shapes

1) INT8_FC
Inputs:
- X: int8[IN]
- W: int8[OUT][IN]
- B: int32[OUT]  (pre-shifted or consistent with SHIFT convention)
Params:
- IN, OUT, SHIFT
Output:
- Y: int8[OUT]

Compute definition
Y[j] = sat8( round_shift( sum_i X[i]*W[j,i] + B[j], SHIFT) )

Expected FPGA mapping (v1 pipelined reference)
- parallel over OUT accumulators
- serial over IN per cycle
- cycles_per_vector ≈ IN
- II_vector ≈ IN (unless double-buffered)

2) INT8_DWCONV1D_K3
Inputs:
- X: int8[C][L]
- W: int8[C][3]
- B: int32[C]
Params:
- C, L, SHIFT
Output:
- Y: int8[C][L-2]

Compute definition
Y[c,t] = sat8( round_shift( X[c,t]*W[c,0] + X[c,t+1]*W[c,1] + X[c,t+2]*W[c,2] + B[c], SHIFT) )

Expected FPGA mapping (v0 sequential reference)
- cycles ≈ C*(L-2)*3
- v1 future: shift-register streaming window -> II=1 per output sample per channel (or small II)
