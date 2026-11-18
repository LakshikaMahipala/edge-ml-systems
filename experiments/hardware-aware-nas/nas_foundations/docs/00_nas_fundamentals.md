# NAS fundamentals 

Neural Architecture Search (NAS) = automatically choose a model architecture from a defined space.

NAS has 3 core pieces:

1) Search space
- what architectures are allowed?
- blocks, widths, depths, kernel sizes, skip connections

2) Search strategy
- how do we explore the space?
- random search, evolutionary, RL controller, Bayesian opt, differentiable (DARTS), one-shot supernet

3) Evaluation / objective
- how do we score candidates?
- accuracy + constraints (latency, params, MACs, memory)

Hidden truth:
Most “NAS papers” differ mainly in evaluation tricks.
The most important engineering decision is the search space + constraint metric.
