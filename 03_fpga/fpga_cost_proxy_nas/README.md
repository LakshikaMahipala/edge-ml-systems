FPGA-aware NAS Cost Proxy 

What exists
- arch → simplified layer sequence
- proxy ops, proxy cycles/LUT/BRAM/BW model
- ranking script: acc_proxy - λ * hw_cost
- join script to compare with reduced-training and zero-cost scores

Run later
python src/rank_with_fpga_cost.py --P 64 --lam_hw 0.25
python src/join_scores.py
