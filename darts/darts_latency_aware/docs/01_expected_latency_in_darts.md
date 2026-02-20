# Expected latency under mixed ops

In DARTS, an edge output is:
y = Σ_k p_k(α) * op_k(x), where p = softmax(α)

We define expected latency as:
E[latency] = Σ_k p_k(α) * latency(op_k)

This is differentiable w.r.t α because p_k depends on α.

Important:
This assumes latency is additive across edges/layers,
which is a good first-order approximation.
