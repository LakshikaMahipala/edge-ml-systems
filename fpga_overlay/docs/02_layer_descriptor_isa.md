# Layer descriptor ISA (what the hardware consumes)

We define a "layer descriptor" as the instruction format for the overlay.

For Conv/BN/ReLU, descriptor fields:

- op: "conv_bn_relu"
- input_addr, output_addr, weight_addr, bn_addr
- Cin, Cout, H, W
- K (kernel), stride, padding
- dtype: int8
- tiling: Tm (out-ch tile), Tn (in-ch tile), Tr, Tc
- activation_scale/zero_point (optional)
- weight_scale/zero_point (optional)

The overlay controller loops:
for m in 0..Cout step Tm:
  for r,c in tiles:
    for n in 0..Cin step Tn:
      load weights tile
      load input tile
      compute MACs → psum
    apply BN + ReLU
    store output tile
