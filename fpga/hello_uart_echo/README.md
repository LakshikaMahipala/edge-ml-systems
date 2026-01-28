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
Hello FPGA — UART Echo (FPGA Day 1)

What this is
- UART RX receives bytes (8N1)
- Echo logic forwards received byte to UART TX
- Testbench sends 3 bytes and checks TX matches exactly

Why this matters
- UART is the simplest realistic host<->FPGA interface.
- This becomes the base transport for later INT8 inference kernels:
  host sends input vector -> FPGA computes -> FPGA returns logits.

Folder structure
- rtl/ : synthesizable UART RX/TX and top
- tb/  : self-checking simulation
- scripts/ : simulation helpers

Run 

Option A: iverilog (fast)
1) Install:
   sudo apt-get install iverilog
2) Run:
   chmod +x scripts/sim_iverilog.sh
   ./scripts/sim_iverilog.sh
Expected output:
- PASS for each byte and ALL PASS

Option B: Vivado xsim
- Follow docs/fpga_setup_vivado.md
- Add rtl/*.v and tb/*.sv to a Vivado project
- Set tb_uart_echo as simulation top
- Run simulation; check PASS in console

Key signals to inspect (waveform)
- rx_valid: pulse when byte is received
- tx_busy: goes high during transmit
- uart_tx_o: serial waveform (start bit low, then data bits, then stop bit high)
