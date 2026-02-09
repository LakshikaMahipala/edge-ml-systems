Mini-project 7: IR -> passes -> FPGA cost 

What it demonstrates
- Take Relay IR text (from tvm_relay_intro)
- Run analysis: op counts
- Run rewrite: toy fusion demo
- Convert to an FPGA op-graph (demo mapping)
- Estimate latency using fpga_cost_model

Run later
./scripts/run_pipeline_later.sh

Key idea
Hardware-aware compilation must connect graph passes to a target cost model.
