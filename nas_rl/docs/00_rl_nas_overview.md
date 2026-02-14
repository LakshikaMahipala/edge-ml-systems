# RL NAS overview 

RL-NAS uses a controller policy πθ to generate an architecture a.
We then evaluate that architecture and get a reward R(a).
The controller updates θ to increase probability of architectures with higher reward.

Pipeline:
controller → sample architecture → evaluate → reward → policy update

Key:
Reward is not just accuracy.
In hardware-aware NAS, reward combines:
- accuracy
- latency/params/MACs/power constraints
