# Discrete → continuous relaxation (the key trick)

Suppose each edge can choose one op from a set:
O = {op1, op2, ..., opK}

In discrete NAS, we pick exactly one:
y = op_j(x)

In DARTS, we define a mixed op:
y = Σ_k softmax(α)_k * op_k(x)

So α are logits. softmax(α) gives mixture weights.

Interpretation:
- early in search: mixture explores many ops
- later: α becomes sharp, one op dominates

Hidden detail:
This is NOT the same as discrete choice:
- mixed op computes all ops and adds them
- gradients flow through all ops
This can bias the search toward certain ops (e.g., skip connections).

End step (discretization):
Pick argmax(α) per edge/op and rebuild a normal discrete network.
