# Binary Neural Network Core: XNOR-Popcount Dot Product in Verilog

> The hardware trick behind binary neural networks: a 32-element multiply-accumulate replaced by one XNOR, one popcount and one add.

**Status:** 🟢 Reference model tested (pytest) · bug fixed in start/first-word handling · random-stream testbench runs in CI

## The idea
In a binary neural network (BNN), weights and activations are only −1 or +1, stored as one bit each (1 → +1, 0 → −1). For two such vectors of length n:

```
dot(a, w) = (# positions that agree) − (# positions that disagree)
          = 2 · popcount( XNOR(a, w) ) − n
```

**Worked example** with 8 elements:

| | bits | values |
|---|---|---|
| a | `1 1 0 1 0 0 1 1` | +1 +1 −1 +1 −1 −1 +1 +1 |
| w | `1 0 0 1 1 0 1 0` | +1 −1 −1 +1 +1 −1 +1 −1 |
| XNOR | `1 0 1 1 0 1 1 0` | popcount = 5 |

dot = 2·5 − 8 = **2**. Multiplying the ±1 values directly gives the same: +1 −1 +1 +1 −1 +1 +1 −1 = 2.

On an FPGA this needs a few LUTs per 32 elements, compared with 32 multipliers (or DSP slices) for INT8. That's why BNNs fit on very small devices.

## Hardware
| Module | Role |
|---|---|
| [`popcount32.v`](rtl/popcount32.v) | 5-level adder tree: 32 bits → 6-bit count |
| [`xnor_popcount_dot.v`](rtl/xnor_popcount_dot.v) | Streams one 32-bit word per cycle, accumulates `2·popcount − 32` |
| [`bnn_dot_top.v`](rtl/bnn_dot_top.v) | Top-level wrapper |

Interface: `start` clears the accumulator, `word_valid` marks an input word, `last_word` marks the final word. `done` and `acc_out` are registered one cycle after the last word. A 256-bit dot product takes 8 cycles.

**Bug fixed:** before, if `start` and the first `word_valid` arrived in the same cycle, the first word was added to the *previous* result instead of to zero. The testbench now deliberately sends them together.

## Verification
- `golden/bnn_ref.py`: bit-exact model, plus a test proving XNOR-popcount equals the plain ±1 dot product (lengths 32–1024)
- `tb/tb_bnn_dot.sv`: 500 random 256-bit vectors plus all-agree (+256) and all-disagree (−256) cases, streamed with random stalls

```bash
make test   # pytest
make sim    # generate vectors + Icarus Verilog simulation
```

## Next steps
- [ ] Multi-neuron binary layer with a sign/threshold activation
- [ ] Synthesis report (LUTs per 32-bit word) compared with an INT8 MAC
- [ ] Wider words (64/128 bits per cycle) for higher throughput
