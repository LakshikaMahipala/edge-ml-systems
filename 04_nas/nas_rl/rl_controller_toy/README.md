Toy RL-NAS Controller 

What this is
- A minimal REINFORCE controller that samples architectures from a discrete search space.
- Uses proxy reward (acc_proxy - penalties) because we cannot train candidates yet.

Run later
Set PYTHONPATH so reward_proxy can import build_model and proxy_metrics:
PYTHONPATH=../nas_foundations/tiny_cnn_search_space/src python src/train_controller.py --steps 200

Outputs
- results/rollouts.jsonl (one JSON per step)
- printed greedy architecture after training

Next
Replace acc_proxy with:
- reduced training accuracy
- measured latency (hardware-aware NAS)
