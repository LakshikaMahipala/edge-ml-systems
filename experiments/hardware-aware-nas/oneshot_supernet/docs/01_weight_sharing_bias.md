# Weight sharing bias (the hidden problem)

In supernet NAS, subnet accuracy is measured using shared weights, not weights trained specifically for that subnet.

This causes ranking problems because:
- subnets interfere with each other during training
- some subnets are sampled more often → their weights are better trained
- larger subnets can dominate training signal unless controlled

Therefore:
shared-weight accuracy is a proxy, not truth.

Mitigations (later weeks):
- uniform sampling (fair training)
- sandwich rule (train smallest + largest + random)
- progressive shrinking (OFA style)
- re-evaluate / aging evaluation
