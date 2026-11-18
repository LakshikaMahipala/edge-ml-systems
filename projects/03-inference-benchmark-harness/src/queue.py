# inference_bench/src/queue.py
from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Sentinel:
    """
    Special marker used to signal shutdown through the queue.
    """
    reason: str = "shutdown"


class BoundedQueue(Generic[T]):
    """
    A thin wrapper around queue.Queue that enforces bounded capacity
    and supports a sentinel-based shutdown protocol.
    """
    def __init__(self, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be > 0")
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)

    def put(self, item: T, timeout: Optional[float] = None) -> None:
        self._q.put(item, timeout=timeout)

    def get(self, timeout: Optional[float] = None) -> T:
        return self._q.get(timeout=timeout)  # type: ignore[return-value]

    def task_done(self) -> None:
        self._q.task_done()

    def join(self) -> None:
        self._q.join()

    def put_sentinel(self, reason: str = "shutdown") -> None:
        self._q.put(Sentinel(reason=reason))

    def is_sentinel(self, item: object) -> bool:
        return isinstance(item, Sentinel)
