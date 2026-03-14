# inference_bench/src/pipeline.py
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, List, Any

from inference_bench.src.queue import BoundedQueue, Sentinel


@dataclass
class PipelineConfig:
    queue_size: int = 8
    max_items: int = 200
    open_loop_fps: float = 0.0  # 0 = closed-loop (backpressure). >0 = fixed-rate producer.
    drop_when_full: bool = False  # if True, producer drops instead of blocking when queue full


@dataclass
class ItemEnvelope:
    item_id: int
    payload: Any
    t_enq: float  # enqueue timestamp (perf_counter)


@dataclass
class ItemTiming:
    item_id: int
    t_enq: float
    t_deq: float
    t_done: float

    @property
    def queue_wait_s(self) -> float:
        return self.t_deq - self.t_enq

    @property
    def service_s(self) -> float:
        return self.t_done - self.t_deq

    @property
    def e2e_s(self) -> float:
        return self.t_done - self.t_enq


@dataclass
class PipelineStats:
    produced: int = 0
    consumed: int = 0
    dropped: int = 0
    wall_time_s: float = 0.0


class ProducerConsumerPipeline:
    """
    Producer: generates payloads (e.g., frames or image paths) and enqueues them.
    Consumer: processes payloads (preprocess -> inference -> postprocess).
    Measures per-item queue wait, service time, and end-to-end time.
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        producer_fn: Callable[[int], Any],
        consumer_fn: Callable[[Any], Any],
    ) -> None:
        self.cfg = cfg
        self.q: BoundedQueue[object] = BoundedQueue(maxsize=cfg.queue_size)
        self.producer_fn = producer_fn
        self.consumer_fn = consumer_fn

        self.stats = PipelineStats()
        self.timings: List[ItemTiming] = []

        self._stop = threading.Event()
        self._t_prod: Optional[threading.Thread] = None
        self._t_cons: Optional[threading.Thread] = None

        self._lock = threading.Lock()

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
        t0 = time.perf_counter()

        period_s = 0.0
        if self.cfg.open_loop_fps > 0:
            period_s = 1.0 / self.cfg.open_loop_fps

        for i in range(self.cfg.max_items):
            if self._stop.is_set():
                break

            payload = self.producer_fn(i)
            env = ItemEnvelope(item_id=i, payload=payload, t_enq=time.perf_counter())

            if self.cfg.drop_when_full:
                # Non-blocking put: if full, drop.
                try:
                    self.q._q.put_nowait(env)  # internal access ok for controlled utility
                    self.stats.produced += 1
                except Exception:
                    self.stats.dropped += 1
            else:
                # Blocking put = backpressure (closed-loop)
                self.q.put(env)
                self.stats.produced += 1

            if period_s > 0:
                # open-loop camera-like pacing
                time.sleep(period_s)

        # Signal shutdown
        self.q.put_sentinel(reason="producer finished")

        t1 = time.perf_counter()
        self.stats.wall_time_s = t1 - t0

    def _consumer_loop(self) -> None:
        while True:
            obj = self.q.get()
            t_deq = time.perf_counter()
            try:
                if isinstance(obj, Sentinel):
                    break

                assert isinstance(obj, ItemEnvelope)
                _ = self.consumer_fn(obj.payload)
                t_done = time.perf_counter()

                with self._lock:
                    self.timings.append(ItemTiming(item_id=obj.item_id, t_enq=obj.t_enq, t_deq=t_deq, t_done=t_done))
                self.stats.consumed += 1

            finally:
                self.q.task_done()
