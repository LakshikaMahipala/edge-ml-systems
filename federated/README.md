Federated 

Goal
Understand FedAvg (federated averaging) via a CPU-only simulator.

Docs
- docs/fedavg_theory.md
- docs/non_iid_notes.md

Scripts
- scripts/fedavg_sim.py            (IID)
- scripts/fedavg_non_iid_demo.py   (non-IID)

Run 
python scripts/fedavg_sim.py --K 10 --n_per 200 --rounds 50 --client_frac 0.5 --local_steps 10 --lr 0.05
python scripts/fedavg_non_iid_demo.py --K 10 --n_per 200 --rounds 50 --client_frac 0.5 --local_steps 10 --lr 0.05
