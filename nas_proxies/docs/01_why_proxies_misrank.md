# Why proxies mis-rank architectures

Mis-ranking causes:
1) Optimization speed differences:
   - some architectures have better early gradients
   - others need more epochs to shine

2) Regularization / capacity interactions:
   - bigger models may overfit late but look good early
   - smaller models may look worse early but generalize later

3) Randomness:
   - init and minibatch noise matter at small budgets

4) Weight-sharing (supernet setting):
   - shared weights bias candidate evaluation
   - ranking drift happens as supernet training progresses

Key result:
Proxy correlation with true accuracy is imperfect.
