# Bilevel optimization (the real DARTS core)

DARTS is written as a bilevel problem:

Inner problem (train weights):
w*(α) = argmin_w  L_train(w, α)

Outer problem (choose architecture using validation):
min_α  L_val(w*(α), α)

Meaning:
- w learns to fit training data, given architecture α
- α learns to generalize (validation loss), knowing that w depends on α

Why we split train/val:
If α is optimized on training loss, it will overfit architecture choice.

Practical training loop:
Repeat:
1) Update w using training minibatch (α fixed)
2) Update α using validation minibatch (w fixed OR approximated)

This alternating loop is what you implement in code.
