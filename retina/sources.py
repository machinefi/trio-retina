"""Frame sources.

A source is just an iterable of `(frame, timestamp)` pairs, so the core never
depends on OpenCV — you can feed it a list of numpy arrays in tests. The
`video_frames` helper is an optional convenience that uses OpenCV (install with
`pip install 'retina-sdk[video]'`) for files / RTSP / webcam.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np


def video_frames(
    source: str | int,
    *,
    stride: int = 1,
    live: bool = False,
    max_frames: int | None = None,
) -> Iterator[tuple[np.ndarray, float]]:
    """Yield (frame, timestamp) from a file path, RTSP/HLS URL, or webcam index.

    `stride` samples every Nth frame (cheap frame-rate reduction). Timestamps are
    media-time (frame_idx / fps) for files, or wall-clock epoch when `live=True`.
    """
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - exercised only with extra
        raise ImportError(
            "video_frames needs OpenCV. Install with: pip install 'retina-sdk[video]'"
        ) from e

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video source: {source!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    emitted = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                ts = time.time() if live else idx / fps
                yield frame, ts
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()
