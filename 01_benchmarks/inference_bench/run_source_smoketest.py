# inference_bench/run_source_smoketest.py
from __future__ import annotations

import argparse

import numpy as np

from inference_bench.src.sources import ImageFolderSource, VideoFileSource


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_folder", type=str, default="")
    ap.add_argument("--video", type=str, default="")
    ap.add_argument("--max_frames", type=int, default=10)
    args = ap.parse_args()

    if not args.image_folder and not args.video:
        raise SystemExit("Provide either --image_folder or --video")

    if args.image_folder:
        src = ImageFolderSource(folder=args.image_folder, max_frames=args.max_frames)
    else:
        src = VideoFileSource(path=args.video, max_frames=args.max_frames)

    count = 0
    for frame in src:
        bgr = frame.bgr
        print(f"Frame {frame.index}: shape={bgr.shape}, dtype={bgr.dtype}, min={int(bgr.min())}, max={int(bgr.max())}")
        count += 1

    print(f"Read {count} frames successfully.")


if __name__ == "__main__":
    main()
