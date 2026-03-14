Pruning Small Model 

Includes
- baseline training
- global unstructured magnitude pruning
- structured neuron pruning (shrinks model)

Run later
python src/train_base.py --epochs 10
python src/prune_unstructured.py --amount 0.5
python src/prune_structured.py --keep_ratio 0.5
