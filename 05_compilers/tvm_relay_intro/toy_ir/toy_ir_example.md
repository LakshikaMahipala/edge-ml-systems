Toy IR example (why IR is useful)

We build a tiny graph:
y = relu(a @ b + c)

Graph representation:
matmul(a,b) -> t1
add(t1,c) -> t2
relu(t2) -> y

Then we do two optimizations:
1) Constant folding: if c is constant, keep it as a constant node.
2) Fusion: fuse add+relu into a single "add_relu" op to reduce memory traffic.

This is the exact kind of reasoning Relay enables at scale.
