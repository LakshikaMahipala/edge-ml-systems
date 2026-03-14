# Memory map (software ↔ overlay contract)

We define a stable memory map so software can feed layers consistently.

Example address spaces:
- DRAM/host memory addresses for tensors:
  input_addr: base pointer to NCHW activations
  output_addr: base pointer
  weight_addr: base pointer to [Cout, Cin, K, K] packed weights
  bn_addr: base pointer to BN params (gamma,beta,mean,var or folded scale/bias)

On-chip BRAM regions (internal, not visible to SW):
- IBuf: input tile buffer
- WBuf: weight tile buffer
- PBuf: partial sums
- OBuf: output tile buffer

Important:
This "contract" is what lets you swap models without rewriting RTL.
