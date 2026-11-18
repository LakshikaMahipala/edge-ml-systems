# Aging evaluation

Aging evaluation = periodically re-evaluate candidates.
You treat proxy scores as "stale".

Procedure:
1) Maintain a pool of candidates with proxy scores.
2) Every K iterations:
   - select top-M candidates
   - re-run proxy training/eval from scratch (or longer budget)
   - update their scores
3) Re-rank based on refreshed scores.

Why it helps:
- corrects early noise
- reduces the chance that one lucky proxy run dominates
- improves stability of selection under limited budget
