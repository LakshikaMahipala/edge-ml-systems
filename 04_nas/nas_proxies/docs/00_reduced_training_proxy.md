# Reduced-training proxies 

Idea:
Instead of fully training every architecture, train each candidate for a small budget:
- few epochs
- small subset of data
- smaller input size

Proxy score:
- validation accuracy after small training budget
- sometimes training loss slope or early accuracy

Why it works:
Good architectures often learn faster early.

Why it fails:
Some architectures learn slowly but surpass later.
So early ranking can be wrong.
