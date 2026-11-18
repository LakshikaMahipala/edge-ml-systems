FPGA Toolchain Decision

Decision
- Selected toolchain: Xilinx Vivado (WebPACK / free edition where applicable)

Reasons
- Strong industry adoption and learning value
- Integrated simulator (xsim) for RTL-level verification
- Clean flow for: RTL -> simulate -> synth -> implement -> bitstream (later when board exists)
- Future extension path to HLS (Vitis HLS) if needed

What we do now (no board)
- Focus on simulation-first deliverables:
  1) UART echo (RTL + testbench)
  2) Fixed-point primitives (coming in Week 5+ FPGA days)
  3) Host/FPGA protocol definition (Week 5–6 FPGA tasks)

What we postpone (needs board)
- Pin constraints (XDC) tied to a specific FPGA board
- Bitstream programming and on-device validation
