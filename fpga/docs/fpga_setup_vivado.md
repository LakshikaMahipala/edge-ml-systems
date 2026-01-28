Vivado Setup

Goal
- Install Vivado and be able to run RTL simulation (xsim).
- Synthesis/programming is postponed until hardware is available.

Install 
- Download Vivado installer (WebPACK / free license path if needed).
- Install with:
  - Vivado
  - Simulator (xsim)

Sanity checks (later)
- Verify vivado launches.
- Verify xsim can run a simple SystemVerilog testbench.

Project in this repo
- fpga/hello_uart_echo
  - rtl/: UART RX/TX + top
  - tb/: self-checking testbench

How to simulate (two options)

Option A: Vivado xsim (recommended when installed)
- Create a Vivado project targeting any Artix-7 device (placeholder).
- Add rtl/*.v and tb/*.sv
- Set tb_uart_echo as top simulation module
- Run "Run Simulation"

Option B: Open-source simulation (fast local-only)
- If you install iverilog + gtkwave:
  iverilog -g2012 -o sim tb/tb_uart_echo.sv rtl/uart_rx.v rtl/uart_tx.v rtl/uart_echo_top.v
  vvp sim

What to record (when you run)
- Testbench should pass with no mismatches.
- Capture waveform screenshot showing:
  - rx_valid pulses
  - tx busy / tx line toggles
