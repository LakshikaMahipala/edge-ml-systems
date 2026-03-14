# Insertion policy (why schedules improve)

When assigning a task to a device, we don't just append at the end.
We try to insert it into the earliest idle gap that fits after dependencies are ready.

This reduces makespan and is part of the standard HEFT behavior.
