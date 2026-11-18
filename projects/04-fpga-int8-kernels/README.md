# INT8 Neural-Network Kernels in SystemVerilog

> Fully-connected and depthwise-convolution layers in INT8 hardware, checked bit-for-bit against a Python golden model on hundreds of random vectors.

**Status:** 🟢 Golden model tested (pytest) · state machines checked cycle-by-cycle in a Python emulation · RTL simulation runs in CI with Icarus Verilog

## What's inside

| Module | What it computes | Hardware cost | Latency |
|---|---|---|---|
| [`fxp_add`, `fxp_mul`](rtl/fxp_ops.sv) | Saturating Q1.7 add and multiply | 1 adder / 1 multiplier | combinational |
| [`int8_fc`](rtl/int8_fc.sv) | `y = requant(W·x + b)`, all outputs in parallel | OUT × IN multipliers | combinational |
| [`int8_fc_seq`](rtl/int8_fc_seq.sv) | Same layer, one input element per cycle | OUT multipliers | IN + 1 cycles |
| [`dwconv1d_int8`](rtl/dwconv1d_int8.sv) | Depthwise 1D convolution (MobileNet-style), one MAC per cycle | 1 multiplier | C · (L−K+1) · (K+1) cycles |
| [`uart/`](uart) | UART echo, framed UART protocol with CRC-8, Python host client | — | — |

`int8_fc` vs `int8_fc_seq` is the classic **area-versus-latency trade-off**: for a 16→8 layer, 128 multipliers give the answer in one cycle, while 8 multipliers need 17 cycles.

## Number format

| Signal | Type |
|---|---|
| inputs `x`, weights `W` | int8 (two's complement) |
| bias `b`, accumulator | int32 |
| output `y` | int8 |

Requantization (shared by all kernels, in [`fxp_pkg.sv`](rtl/fxp_pkg.sv)):

```
y = clip( (acc + 2^(SHIFT-1)) >>> SHIFT , -128, 127 )
```

**Worked example** (SHIFT = 8): acc = 1000 → (1000 + 128) >>> 8 = 1128 / 256 = 4.4 → **4**. And acc = 40000 → 156.75 → clipped to **127**.

[`golden/int8_ref.py`](golden/int8_ref.py) also shows how a *float* layer becomes these integers (`quantize_fc_layer`): symmetric weight quantization, bias stored in accumulator units, and the smallest power-of-two shift that avoids saturation.

## Verification flow

```
golden/int8_ref.py ──► golden/gen_vectors.py ──► vectors/*.hex + config.svh
   (bit-exact model)      (200 random vectors,          │
   pytest: 10 tests        first two all-127/-128)      ▼
                                              tb/*.sv  ($readmemh, compare every output,
                                                        and check cycle counts)
```
- `tb_fxp_ops`: **exhaustive**, all 65,536 int8 input pairs for add and multiply
- `tb_int8_fc`, `tb_int8_fc_seq`: 200 vectors × 8 outputs; the sequential version also checks latency = IN + 1
- `tb_dwconv1d_int8`: 200 vectors × 120 outputs, plus the exact cycle count
- Sizes come from the generated `config.svh`, so the RTL, testbench and model can't disagree.

## Run it
```bash
pip install numpy pytest
sudo apt install iverilog        # Icarus Verilog >= 11
make test     # golden-model unit tests
make sim      # generate vectors and run all four testbenches
```

## Layout
```
rtl/      fxp_pkg.sv · fxp_ops.sv · int8_fc.sv · int8_fc_seq.sv · dwconv1d_int8.sv
tb/       one self-checking testbench per module
golden/   int8_ref.py (reference) · gen_vectors.py · test_int8_ref.py
uart/     uart_echo · uart_frame_protocol (CRC-8 framing) · host_tools (Python client)
docs/     fixed-point & requantization notes · depthwise vs pointwise · latency/II notes
```

## Next steps
- [ ] Synthesize for an FPGA (e.g. Artix-7) and report LUT / DSP / BRAM use and Fmax
- [ ] Connect `int8_fc_seq` to the UART frame receiver for on-board inference from Python
- [ ] Double-buffered input so the next vector loads while the current one computes (II < latency)
