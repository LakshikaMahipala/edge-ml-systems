# HEFT intuition

We have a DAG of tasks (ops), and multiple devices (CPU, GPU, FPGA).
Each task has different runtime on each device.
Tasks have dependencies and may require communication if placed on different devices.

Goal:
Minimize total completion time (makespan).

HEFT idea:
1) Give each task a priority (upward rank).
2) Schedule tasks in priority order.
3) For each task, pick the device that yields earliest finish time,
   using an insertion policy that can place tasks into idle slots.
