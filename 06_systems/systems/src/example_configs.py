from __future__ import annotations
from transfer_models import Link

PCIE_GEN4_X16 = Link("pcie_gen4_x16_effective", bandwidth_GBps=22.0, overhead_us=15.0)
ETH_100G_RDMA = Link("100g_rdma_effective", bandwidth_GBps=10.0, overhead_us=50.0)
ETH_10G_RPC   = Link("10g_rpc_effective", bandwidth_GBps=1.0, overhead_us=200.0)
UART_1MBps    = Link("uart_1MBps", bandwidth_GBps=0.001, overhead_us=500.0)
