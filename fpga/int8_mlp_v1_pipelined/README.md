INT8 MLP v1 (pipelined) 

What changed from v0
- v0: combinational dot product (correctness-first)
- v1: sequential MAC engine (1 input element per cycle) with OUT-parallel accumulators

Performance model
- cycles_per_vector ≈ IN
- latency_first_output ≈ IN cycles
- II_vector ≈ IN (single-buffer)

Run (later)
./scripts/sim_iverilog.sh
Expected: PASS tb_int8_fc_pipelined
