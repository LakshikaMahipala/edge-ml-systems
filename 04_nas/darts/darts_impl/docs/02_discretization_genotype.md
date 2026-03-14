# Discretization (genotype extraction)

After search, each MixedOp has α logits over candidate ops.

We discretize:
- pick op = argmax(softmax(α))
- store a genotype listing chosen ops per edge/node

Then:
- rebuild a normal discrete network
- train from scratch for honest evaluation

In our repo today:
- we implement genotype extraction + JSON export
- rebuilding discrete net comes later (Mini-project 12)
