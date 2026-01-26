# inference_bench/src/sources.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Frame:
    """
    Raw frame container (BGR uint8 by OpenCV convention).
    """
    index: int
    bgr: np.ndarray  # HWC uint8 BGR


class FrameSource:
    def __iter__(self) -> Iterator[Frame]:
        raise NotImplementedError


@dataclass
class ImageFolderSource(FrameSource):
    folder: str
    max_frames: Optional[int] = None
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")

    def __post_init__(self) -> None:
        p = Path(self.folder)
        if not p.exists():
            raise FileNotFoundError(f"Folder not found: {p}")
        files: List[Path] = []
        for ext in self.extensions:
            files.extend(sorted(p.glob(f"*{ext}")))
        self._files = files
        if self.max_frames is not None:
            self._files = self._files[: self.max_frames]

    def __iter__(self) -> Iterator[Frame]:
        for i, fp in enumerate(self._files):
            img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if img is None:
                continue
            yield Frame(index=i, bgr=img)


@dataclass
class VideoFileSource(FrameSource):
    path: str
    max_frames: Optional[int] = None

    def __post_init__(self) -> None:
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"Video file not found: {p}")

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.path}")

        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield Frame(index=i, bgr=frame)
            i += 1
            if self.max_frames is not None and i >= self.max_frames:
                break

        cap.release()
