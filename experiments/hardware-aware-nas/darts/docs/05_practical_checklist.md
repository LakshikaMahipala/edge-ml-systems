# DARTS practical checklist (what you do in real projects)

Before running:
- define search space (ops per edge)
- define macro-cell / supernet skeleton
- split train/val properly
- make α parameters separate from w parameters

During search:
- alternate: update w on train, α on val
- log α distributions over time (detect collapse)
- monitor if skips dominate
- keep search budget small first (tiny run)

After search:
- discretize: argmax α per edge/op
- rebuild discrete network
- train from scratch and compare to baseline

Hardware-aware next step:
- add latency penalty term to L_val 
