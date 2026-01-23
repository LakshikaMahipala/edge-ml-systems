# inference_bench/src/pipeline.py
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from inference_bench.src.queue import BoundedQueue, Sentinel


@dataclass
class PipelineConfig:
    queue_size: int = 8
    num_items: int = 200  # number of items to process in demo
    producer_sleep_ms: float = 0.0  # simulate data arrival time
    consumer_sleep_ms: float = 0.0  # simulate output handling time


@dataclass
class PipelineStats:
    produced: int = 0
    consumed: int = 0
    dropped: int = 0
    wall_time_s: float = 0.0


class ProducerConsumerPipeline:
    """
    Producer: generates items (e.g., image frames), optionally pre-processes.
    Consumer: runs inference + postprocess.
    Uses bounded queue to avoid runaway latency.
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        producer_fn: Callable[[int], object],
        consumer_fn: Callable[[object], object],
    ) -> None:
        self.cfg = cfg
        self.q: BoundedQueue[object] = BoundedQueue(maxsize=cfg.queue_size)
        self.producer_fn = producer_fn
        self.consumer_fn = consumer_fn
        self.stats = PipelineStats()
        self._stop = threading.Event()
        self._t_prod: Optional[threading.Thread] = None
        self._t_cons: Optional[threading.Thread] = None

    def start(self) -> None:
        self._t_prod = threading.Thread(target=self._producer_loop, name="producer", daemon=True)
        self._t_cons = threading.Thread(target=self._consumer_loop, name="consumer", daemon=True)
        self._t_prod.start()
        self._t_cons.start()

    def stop(self, reason: str = "stop requested") -> None:
        self._stop.set()
        self.q.put_sentinel(reason=reason)

    def join(self) -> None:
        if self._t_prod is not None:
            self._t_prod.join()
        if self._t_cons is not None:
            self._t_cons.join()

    def _producer_loop(self) -> None:
        for i in range(self.cfg.num_items):
            if self._stop.is_set():
                break

            item = self.producer_fn(i)

            # Bounded queue: blocks when full -> this is BACKPRESSURE.
            # Backpressure is how we prevent p99 latency explosion.
            self.q.put(item)
            self.stats.produced += 1

            if self.cfg.producer_sleep_ms > 0:
                time.sleep(self.cfg.producer_sleep_ms / 1000.0)

        # Signal consumer shutdown
        self.q.put_sentinel(reason="producer finished")

    def _consumer_loop(self) -> None:
        while True:
            item = self.q.get()
            try:
                if isinstance(item, Sentinel):
                    break

                _ = self.consumer_fn(item)
                self.stats.consumed += 1

                if self.cfg.consumer_sleep_ms > 0:
                    time.sleep(self.cfg.consumer_sleep_ms / 1000.0)

            finally:
                # In this simple pipeline, we call task_done for every get().
                self.q.task_done()
