Concurrency Notes 

Why we pipeline
A real ML system is a DAG:
- input capture/IO
- preprocess
- inference
- postprocess
- output/serialization       

If you run these sequentially on one thread:
- CPU and accelerator spend time idle         
- throughput is lower
- tail latency becomes fragile under load

Producer/Consumer pattern
- Producer generates work and pushes into a bounded queue.
- Consumer pulls work and processes it.

Backpressure (critical)
We use a bounded queue so the producer blocks when the queue is full.
This prevents:
- unbounded queue growth
- extreme p99 latency due to long waiting times     

Shutdown correctness
A sentinel object is pushed to signal shutdown.           
This avoids deadlocks where the consumer waits forever.

Common pitfalls
- Unbounded queues (p99 explodes under load)
- Large object copies through the queue (memory pressure)
- No clear shutdown protocol (threads hang)
- Measuring throughput without considering latency SLAs

Next step
Replace “item id” with “preprocessed tensor” so:
- producer: read/decode/normalize (Python now, C++ later)
- consumer: inference/postprocess
Then measure p50/p99 end-to-end and throughput under load.
