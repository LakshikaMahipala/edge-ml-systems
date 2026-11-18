# Dataset format

We store each labeled graph as JSONL:
{
  "graph_id": "...",
  "nodes": [...],
  "edges": [...],
  "globals": {...},
  "target": {"latency_p50_ms": 1.23}
}

We also store pairwise ranking samples as JSONL:
{
  "a_id": "...",
  "b_id": "...",
  "label": 1
}
label=1 means A is slower than B (or define consistently).
We will define it explicitly in code.
