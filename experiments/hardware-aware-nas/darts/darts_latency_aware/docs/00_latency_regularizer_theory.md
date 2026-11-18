# Latency-aware DARTS 

In normal DARTS, the architecture parameters α are trained to minimize validation loss:
min_α L_val(w*(α), α)

To make it hardware-aware, we add a penalty:
min_α [ L_val(w, α) + λ * Latency(α) ]

λ controls the trade-off:
- λ = 0: accuracy only
- larger λ: prefers faster / cheaper ops

This produces architectures on the accuracy–latency Pareto frontier.
