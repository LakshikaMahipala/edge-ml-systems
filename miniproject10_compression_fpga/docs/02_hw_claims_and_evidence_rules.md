# Hardware claims & evidence rules

Rule 1: No speedup claim without measurement context.
Always specify:
- device (CPU/GPU/FPGA)
- precision (FP32/FP16/INT8/1-bit)
- IO regime (UART vs PCIe/Ethernet)
- batch size and input size

Rule 2: Unstructured pruning ≠ speedup by default.
It only speeds up if sparse kernels exist.

Rule 3: FPGA kernels require bit-exact verification.
- Python golden model must match RTL outputs exactly (integer domain)

Rule 4: If results are not run yet:
- present as "planned measurement" with templates
- do not present as achieved

This keeps the repo honest and professional.
