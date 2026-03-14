Embedded Linux I/O Notes

Why I/O matters in inference systems
End-to-end latency includes:
- reading input (disk/network/camera)
- decoding (JPEG/video)
- preprocessing
- inference
- postprocessing
If you measure only inference, you will ship a system that misses deadlines.

Page cache effect
Linux caches file data in RAM. The first read of a file may be slow (cold cache),
later reads may be fast (warm cache). Benchmarks must disclose which mode is used.

Tail latency (p99)
Even if average throughput is high, occasional slow reads can break real-time performance.
p99 is a practical proxy for worst-case behavior.

Chunk size and syscall overhead
Small chunk sizes:
- more syscalls
- more overhead
Large chunk sizes:
- fewer syscalls, generally higher throughput
But too large can interact with memory pressure.

Dropping caches
True cold-cache benchmarking may require:
- root privileges: echo 3 > /proc/sys/vm/drop_caches
Not always possible. If you cannot drop caches, record that limitation honestly.

How this connects to the project
Week 3 I/O benchmarks become part of the latency budget alongside preprocess/inference/postprocess.
Later, we will connect the I/O source to the producer/consumer pipeline.
