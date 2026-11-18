Expected results (before real runs)

Op count
- ResNet will show many conv2d + relu + add
- MobileNet will show depthwise conv + pointwise conv patterns

Cost model
- Under UART, T_io dominates by orders of magnitude for even moderate tensors.
- Predicted compute time is microseconds, predicted I/O is milliseconds.

Conclusion
- HW-aware compilation must include interface modeling.
- FPGA acceleration claims depend on I/O system design.
