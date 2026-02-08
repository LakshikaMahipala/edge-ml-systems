import time
import numpy as np
import cv2

class PerfOverlay:
    def __init__(self, ema_alpha: float = 0.1, max_hist: int = 200):
        self.ema_alpha = ema_alpha
        self.max_hist = max_hist
        self.last_t = None
        self.fps_ema = None
        self.lat_hist = []

    def update(self, latency_ms: float):
        now = time.perf_counter()
        if self.last_t is None:
            self.last_t = now
            return

        dt = now - self.last_t
        self.last_t = now
        fps = 1.0 / max(dt, 1e-9)

        if self.fps_ema is None:
            self.fps_ema = fps
        else:
            self.fps_ema = (1 - self.ema_alpha) * self.fps_ema + self.ema_alpha * fps

        self.lat_hist.append(latency_ms)
        if len(self.lat_hist) > self.max_hist:
            self.lat_hist.pop(0)

    def stats(self):
        if not self.lat_hist:
            return None
        arr = np.array(self.lat_hist, dtype=np.float32)
        return {
            "fps_ema": float(self.fps_ema) if self.fps_ema is not None else 0.0,
            "lat_p50": float(np.percentile(arr, 50)),
            "lat_p99": float(np.percentile(arr, 99)),
        }

    def draw(self, frame_bgr, top1: int, conf: float, latency_ms: float):
        self.update(latency_ms)
        st = self.stats() or {"fps_ema": 0.0, "lat_p50": 0.0, "lat_p99": 0.0}

        lines = [
            f"top1={top1} conf={conf:.3f}",
            f"lat={latency_ms:.2f} ms (p50={st['lat_p50']:.2f}, p99={st['lat_p99']:.2f})",
            f"fps(ema)={st['fps_ema']:.2f}",
        ]

        y = 30
        for s in lines:
            cv2.putText(frame_bgr, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            y += 30
        return frame_bgr
