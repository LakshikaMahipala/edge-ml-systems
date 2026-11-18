# Training loop explained (first-order DARTS)

We keep two parameter sets:
- w: normal weights (conv kernels, batchnorm, classifier)
- α: architecture parameters (logits for choosing ops inside MixedOp)

Loop:
repeat for steps:
1) w-step (train batch):
   w ← w - ηw ∇w L_train(w, α)

2) α-step (val batch):
   α ← α - ηα ∇α L_val(w, α)
(first-order: treat w as constant during α update)

This is the simplest bilevel approximation.
