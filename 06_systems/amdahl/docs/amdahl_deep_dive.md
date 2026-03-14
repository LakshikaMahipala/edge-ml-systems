Amdahl’s Law (deep dive) 

Problem
When we speed up one part of a pipeline (e.g., inference), total end-to-end speedup is limited by everything we did NOT speed up.

Classic Amdahl’s Law
Let:
- T_total = total time per item
- T_accel = time of the part we can accelerate (e.g., inference)
- T_rest  = everything else (pre, post, copies, I/O, Python overhead, etc.)

Define:
p = T_accel / T_total   (fraction of time spent in the acceleratable part)
(1-p) = T_rest / T_total (serial fraction w.r.t. that accelerator)

If we speed up the acceleratable part by factor S, new time is:
T_new = T_rest + (T_accel / S)

Speedup = T_total / T_new
       = 1 / ( (1-p) + p/S )

The theoretical maximum (as S→∞):
Speedup_max = 1 / (1-p)

Interpretation (the hidden systems skill)
- If inference is only 30% of your end-to-end time (p=0.3),
  then even infinite inference speed gives max speedup = 1/(0.7)=1.43×.
- That means “FPGA/GPU acceleration won’t move the needle” until we reduce pre/post/copies.

What we do in this repo
We always separate timing into:
- preprocess
- inference
- postprocess
- copies (if any)
Then we run:
- amdahl/scripts/amdahl_speedup.py
to compute:
- p (acceleratable fraction)
- max speedup
- speedup for specific S values (e.g., TRT FP16 2×, INT8 3×, FPGA 5×)

Also note
Amdahl is per-item latency-focused. For throughput with pipelining, bottleneck analysis also matters.
But Amdahl remains the correct guardrail for “will acceleration help end-to-end latency?”.
