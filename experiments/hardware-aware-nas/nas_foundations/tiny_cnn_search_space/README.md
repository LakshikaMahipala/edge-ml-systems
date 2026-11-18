Tiny CNN Search Space 

What exists
- a discrete search space (search_space.py)
- architecture encoding + stable ids
- a model builder that instantiates TinyNet from an arch dict
- proxy metrics: params and rough MACs

Run later
python src/random_sampler.py --n 50 --out results/candidates.jsonl
(then later we will score candidates and train a few)
