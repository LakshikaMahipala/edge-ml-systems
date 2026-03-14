from __future__ import annotations

import argparse
import platform
import sys
import time

import cv2
import numpy as np


def pct_ms(samples_s, p: float) -> float:
    arr = np.array(samples_s, dtype=np.float64) * 1000.0
    return float(np.percentile(arr, p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {args.video}")

    # Warmup (not timed)
    for _ in range(args.warmup):
        ok, _ = cap.read()
        if not ok:
            break

    times = []
    count = 0

    for _ in range(args.frames):
        t0 = time.perf_counter()
        ok, frame = cap.read()
        t1 = time.perf_counter()
        if not ok:
            break
        _ = int(frame[0, 0, 0])  # touch pixel to avoid overly lazy paths
        times.append(t1 - t0)
        count += 1

    cap.release()

    if count == 0:
        raise SystemExit("No frames decoded. Check codec support and file path.")

    mean_ms = float(np.mean(times) * 1000.0)
    p50_ms = pct_ms(times, 50)
    p99_ms = pct_ms(times, 99)
    fps = count / max(float(np.sum(times)), 1e-12)

    print("Video Decode Benchmark (Week 3 Day 5)")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"Video: {args.video}")
    print(f"Frames decoded: {count} (warmup={args.warmup})")
    print("")
    print(f"Decode latency p50:  {p50_ms:.3f} ms")
    print(f"Decode latency p99:  {p99_ms:.3f} ms")
    print(f"Decode latency mean: {mean_ms:.3f} ms")
    print(f"Approx decode FPS:   {fps:.2f}")
    print("")
    print("Notes:")
    print("- Measures cv2.VideoCapture.read() time per frame (decode + copy).")
    print("- p99 >> p50 indicates tail latency risk.")


if __name__ == "__main__":
    main()
