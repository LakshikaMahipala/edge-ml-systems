# Code map 

Core objects:
- ops.py: candidate operations (3x3 conv, 5x5 conv, skip)
- mixed_op.py: softmax-weighted mixture over ops (architecture parameters α live here)
- cell.py: a small directed acyclic graph (DAG) of mixed ops
- supernet.py: stacks cells into a tiny classifier
- darts_trainer.py: alternating optimization of w (train) and α (val) [first-order]
- discretize.py: converts learned α into discrete genotype
- data_stub.py: synthetic dataset placeholder (replace with CIFAR10 later)
- results/: logs of α and derived genotype
