# Decision Matrix (what to use when)

| Goal | Best tool first | Then | Why |
|---|---|---|---|
| Filter huge candidate pool fast | Zero-cost (SNIP/GradNorm) | Reduced-training | One minibatch is cheapest; training confirms |
| Get shortlist likely to train well | Reduced-training proxy | Full training (later) | Captures learnability |
| Enforce FPGA feasibility | FPGA cost proxy | Measure kernel points (later) | Prevents hardware-infeasible picks |
| Automated exploration | RL controller | Replace reward with real metrics | Controller is only as good as reward |
| Trustworthy final choice | Full training + measured latency | — | Only real evidence |

Notes:
- proxies are ranking tools, not truth.
- hardware proxies must be calibrated with measurement later.
