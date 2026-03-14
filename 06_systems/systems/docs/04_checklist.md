# Checklist

When analyzing an accelerator attachment decision, always record:
- payload size per inference (bytes)
- batch size
- bandwidth (effective, not theoretical)
- overhead per request (fixed)
- compute time
- p50 and p99 targets

Then compute:
- transfer fraction = (H2D + D2H) / total
If transfer fraction > 0.5, you are transfer-dominated.
