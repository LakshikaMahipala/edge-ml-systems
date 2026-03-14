PCIe vs Network-attached accelerators

This module covers:
- attachment models (PCIe vs network-attached)
- end-to-end latency budgeting (copy + overhead + compute)
- decision guide for when pooling/network-attached makes sense

Run later:
python src/budget_calculator.py --link pcie --payload_kb 256 --compute_us 2000
python src/budget_calculator.py --link rdma100g --payload_kb 256 --compute_us 2000
python src/budget_calculator.py --link uart --payload_kb 256 --compute_us 2000
