from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from queue import Queue, Full, Empty
from typing import Optional, Dict, Tuple

import serial  # requires pyserial later

from host_tools.protocol import Frame, decode_stream, TYPE_INPUT, TYPE_OUTPUT, TYPE_PING, TYPE_PONG


@dataclass
class AsyncStats:
    crc_fail: int = 0
    frames_rx: int = 0
    frames_tx: int = 0
    timeouts: int = 0
    dropped_rx: int = 0


class UARTAsyncClient:
    """
    Async UART client with:
    - reader thread draining UART
    - decoder and response routing
    - bounded response queue (back-pressure)
    """

    def __init__(
        self,
        port: str,
        baud: int = 115200,
        rx_queue_max: int = 1024,
        read_chunk: int = 4096,
        timeout_s: float = 0.1,
    ):
        self.ser = serial.Serial(port, baud, timeout=timeout_s)
        self.read_chunk = read_chunk

        self._stop = threading.Event()
        self._buf = b""

        # responses keyed by req_id
        self._resp_map: Dict[int, Queue] = {}
        self._lock = threading.Lock()

        # global RX queue for unsolicited frames (optional)
        self.rx_queue: Queue[Frame] = Queue(maxsize=rx_queue_max)

        self.stats = AsyncStats()
        self._thr = threading.Thread(target=self._reader_loop, daemon=True)
        self._thr.start()

    def close(self):
        self._stop.set()
        try:
            self._thr.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.ser.close()
        except Exception:
            pass

    def _reader_loop(self):
        while not self._stop.is_set():
            try:
                chunk = self.ser.read(self.read_chunk)
                if not chunk:
                    continue
                self._buf += chunk
                frames, rem = decode_stream(self._buf)
                self._buf = rem

                for fr in frames:
                    self.stats.frames_rx += 1
                    self._route_frame(fr)
            except Exception:
                # If serial glitches, loop continues; we count as drop-ish
                self.stats.dropped_rx += 1

    def _route_frame(self, fr: Frame):
        # route output frames by req_id if present
        if fr.typ == TYPE_OUTPUT and len(fr.payload) >= 1:
            req_id = fr.payload[0]
            with self._lock:
                q = self._resp_map.get(req_id)
            if q is not None:
                try:
                    q.put_nowait(fr)
                    return
                except Full:
                    self.stats.dropped_rx += 1
                    return

        # otherwise push to global rx_queue (best-effort)
        try:
            self.rx_queue.put_nowait(fr)
        except Full:
            self.stats.dropped_rx += 1

    def send_frame(self, fr: Frame) -> None:
        b = fr.encode()
        self.ser.write(b)
        self.stats.frames_tx += 1

    def ping(self, timeout_s: float = 0.5) -> bool:
        self.send_frame(Frame(typ=TYPE_PING, payload=b""))
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            try:
                fr = self.rx_queue.get(timeout=0.05)
                if fr.typ == TYPE_PONG:
                    return True
            except Empty:
                pass
        self.stats.timeouts += 1
        return False

    def request_vector(
        self,
        req_id: int,
        x_int8: bytes,
        timeout_s: float = 0.5
    ) -> Optional[Tuple[int, bytes, float]]:
        """
        Send INPUT_VECTOR with payload [req_id][x...]
        Await OUTPUT_VECTOR with matching req_id.
        Returns (req_id, y_payload_without_id, latency_ms) or None if timeout.
        """
        if not (0 <= req_id <= 255):
            raise ValueError("req_id must fit in 1 byte")
        payload = bytes([req_id]) + x_int8

        q = Queue(maxsize=1)
        with self._lock:
            self._resp_map[req_id] = q

        t0 = time.perf_counter()
        self.send_frame(Frame(typ=TYPE_INPUT, payload=payload))

        try:
            fr = q.get(timeout=timeout_s)
            t1 = time.perf_counter()
            # cleanup
            with self._lock:
                self._resp_map.pop(req_id, None)
            y = fr.payload[1:]  # strip req_id
            return req_id, y, (t1 - t0) * 1e3
        except Empty:
            self.stats.timeouts += 1
            with self._lock:
                self._resp_map.pop(req_id, None)
            return None
