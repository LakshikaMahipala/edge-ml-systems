Camera/Video Stack Notes

End-to-end inference latency includes:
1) input read/capture (camera driver or disk I/O)
2) decode (JPEG decode or H.264 decode)
3) colorspace conversion (BGR<->RGB)
4) resize + normalize (preprocess)
5) inference
6) postprocess + output packaging

Why decode matters
cv2.VideoCapture.read() includes demux + decode + copy into a frame buffer.
For small models or batch=1, decode can dominate total latency.

What to measure
A) Per-frame decode latency (p50/p99)
B) Decode throughput (approx FPS)
C) Tail latency spikes (p99 >> p50)

How this repo supports it
- VideoFileSource reads frames via OpenCV
- run_video_decode_benchmark.py measures decode time per frame

Interpretation
- If decode FPS < target FPS, real-time is impossible even if inference is fast
- If p99 decode is high, you will see frame drops or latency spikes in streaming systems
