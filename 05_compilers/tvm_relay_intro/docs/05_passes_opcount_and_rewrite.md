Relay passes: analysis vs rewrite 

What a “pass” is
A pass is a transformation or analysis run over an IR.
- analysis pass: gathers facts (op counts, shapes, constants, memory)
- rewrite pass: changes the IR (fusion, layout change, quantization rewrite)

Why ML compilers use passes
Neural nets are graphs. Performance comes from:
- reducing memory traffic (fusion)
- improving locality (layout transforms)
- selecting better kernels (lowering + schedules)
All are driven by passes.

Two passes we practice today (toy versions)
1) Op counting
- read Relay text, count occurrence of key ops:
  nn.conv2d, nn.dense, nn.relu, add, multiply, etc.
- produce a histogram + “top ops” list

2) Pattern rewrite (toy)
- example: rewrite "add then relu" into "add_relu"
- We do this on text to teach the idea.
Real TVM uses graph pattern matchers, but the concept is the same.

How this connects to FPGA cost models
Once we know:
- how many times each op appears
- tensor shapes
We can estimate:
- MACs
- bytes moved
- expected latency using FPGA timing model hooks
Then we can guide compiler decisions:
- prefer blocks we can accelerate
- avoid unsupported ops
