FPGA INT8 MLP v0 — Demo Notes 

What exists now (✅ completed)
1) INT8 FC kernel v0 (fpga/int8_mlp_v0)
- Implements y = requant( dot(x,w) + b )
- int8 inputs + int8 weights
- int32 accumulator
- requant: right shift + rounding + saturate to int8
- Simulation-only for now (iverilog)

2) UART protocol layer (fpga/protocol_uart)
- Robust frame format: SOF + LEN + TYPE + PAYLOAD + CRC8
- Self-checking loopback TB validates:
  - decode correctness
  - CRC integrity
  - re-encode matches original byte stream

3) Host tools scaffold (host_tools)
- Python encoder/decoder for UART frames
- uart_client scaffold (ping + send_vector), to run later with pyserial

What is NOT connected yet (next FPGA step)
- The UART protocol is not yet wired to the int8_fc kernel.
The missing integration is:
UART RX bytes -> frame_rx -> when TYPE=INPUT_VECTOR -> feed x[] -> compute y[] -> frame_tx(TYPE=OUTPUT_VECTOR) -> UART TX bytes

Why we did it this way
- Protocol correctness is separated from kernel correctness.
- This prevents debugging chaos when hardware is introduced.

What we will measure later
- End-to-end latency:
  host send vector -> fpga compute -> host receives logits
- Then compare to CPU/GPU versions using our latency budget model.

Run later
- protocol_uart sim script
- int8_mlp_v0 sim script
Expected outputs:
- PASS tb_uart_frame_loopback
- PASS tb_int8_fc
