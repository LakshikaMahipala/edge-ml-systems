Non-IID data in FL (quick intuition)

IID means every client looks like a random sample of the whole population.
Non-IID means each client is biased (e.g., one user only takes photos of cats).

Effects
- Local SGD pushes weights toward each client’s local optimum.
- When the server averages, updates can partially cancel.
- Convergence slows; final model may be worse.

Typical fixes (not implemented today)
- More frequent aggregation (smaller local epochs E)
- Personalized FL (client-specific heads)
- Regularization / proximal terms (FedProx)
- Better client sampling
