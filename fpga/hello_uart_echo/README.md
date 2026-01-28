Hello FPGA — UART Echo (Simulation-First)

What this is
- A minimal UART loopback system:
  - Receives bytes on RX
  - Immediately transmits same byte on TX

Why it matters
- UART is the simplest realistic host<->FPGA interface.
- This becomes your base transport for later INT8 inference kernels.

Files
- rtl/uart_rx.v
- rtl/uart_tx.v
- rtl/uart_echo_top.v
- tb/tb_uart_echo.sv

Run later
- Use Vivado xsim or iverilog (see docs/fpga_setup_vivado.md)
