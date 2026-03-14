# SNIP-like proxy

Idea (saliency):
If removing a weight would change the loss a lot, that weight is "important".

Classic SNIP uses:
score = sum_i | w_i * dL/dw_i |

We compute this at initialization with one minibatch.

Interpretation:
High saliency suggests the network has many effective parameters
that immediately influence the loss.
