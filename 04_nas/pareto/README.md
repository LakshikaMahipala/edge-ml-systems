Multi-objective optimization + Pareto frontier

Key idea:
There is no single best model under accuracy/latency/energy.
We report the Pareto frontier (non-dominated set).

Code:
- src/pareto.py: dominance + pareto_front
- demos: synthetic candidates + pareto filtering

Run later:
python src/demo_pareto_filter_stub.py
