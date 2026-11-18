DLA debug checklist (what to do when DLA doesn’t work)

Symptoms
1) Engine build fails when targeting DLA
2) Engine builds but runs mostly on GPU
3) Engine runs on DLA but is slower than GPU

Checklist
A) Confirm platform support
- Confirm Jetson module has DLA (Orin/Xavier family have DLA cores; check platform docs).

B) Start with DLA + GPU fallback
- If build fails even with fallback, something is deeply incompatible.

C) Use verbose logs
- Identify which layer causes fallback.
- Common causes: unsupported op, unsupported kernel size/stride/padding, unsupported reshape patterns, dynamic shapes.

D) Try DLA-only as a “compatibility test”
- If it fails, you now know full offload is not possible without model changes.

E) Performance sanity
- DLA is not guaranteed faster than GPU.
- For small batch sizes, GPU often wins.
- Use your copy-time + latency budget tools to decide.

F) Record findings
- Store: build logs, p50/p99, and “layer placement summary”.
