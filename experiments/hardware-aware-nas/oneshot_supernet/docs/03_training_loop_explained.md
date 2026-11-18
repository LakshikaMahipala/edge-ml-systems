# Training loop (supernet)

Each step:
1) Sample a subnet configuration c
2) Activate subnet c inside the supernet
3) Forward/backward only through the active path
4) Update shared weights w

Objective:
min_w E_{c ~ p(c)} [ L_train(w, c) ]

There is no α gradient here — the architecture is sampled discretely.
