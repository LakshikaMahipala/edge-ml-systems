# First-order vs second-order DARTS (why it's a big deal)

When we update α, ideally we need the gradient:
∇_α L_val(w*(α), α)

But w*(α) depends on α, so the true gradient includes a second-order term:
∂L_val/∂w * ∂w*/∂α

Second-order DARTS approximates this using an unrolled SGD step:
w' = w - η ∇_w L_train(w, α)
then use w' inside L_val

This is more accurate but expensive.

First-order DARTS ignores dependence of w on α:
treat w as constant for the α update

Tradeoff:
- First-order: cheaper, less accurate, sometimes more stable
- Second-order: closer to the true bilevel gradient, but can be unstable and heavy

In our repo plan:
- Day 2: implement first-order DARTS first (tiny run)
- optionally add second-order later if needed
