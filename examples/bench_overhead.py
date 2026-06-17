"""Honest micro-benchmark: the Retina-LAYER overhead, detector EXCLUDED.

Edge/robotics users want to know what Retina itself costs *on top of* their
detector — the tracker (IoU association) + rules (zone/line/count) + event
construction per frame. The detector (YOLO, a VLM, …) dominates real latency and
varies by hardware, so we exclude it: detections are synthetic and precomputed,
and we subtract the trivial detector-call time from the full-pipeline time.

Runs with numpy only — no camera, no model, no GPU:

    python examples/bench_overhead.py
    python examples/bench_overhead.py --frames 5000 --tracks 50
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from retina import CallableDetector, CountRule, IoUTracker, Line, LineRule, Retina, Zone, ZoneRule
from retina.detect import Detection


def _make_detections(k: int, frame_idx: int, w: int, h: int) -> list[Detection]:
    """k moving boxes; each drifts right one step per frame so the tracker has to
    do real IoU association frame-to-frame (not a degenerate static scene)."""
    out = []
    for j in range(k):
        cx = (frame_idx * 3 + j * 37) % w
        cy = (j * 53) % h
        out.append(
            Detection(label="person", bbox=(cx - 8, cy - 8, cx + 8, cy + 8), confidence=0.9)
        )
    return out


def _build_retina(k: int, w: int, h: int):
    """A representative pipeline: tracker + a zone, a line, and a count rule."""
    zone = Zone("z", [(0.3, 0.0), (0.7, 0.0), (0.7, 1.0), (0.3, 1.0)], normalized=True)
    line = Line("door", (0.5, 0.0), (0.5, 1.0), normalized=True)
    precomputed: dict[int, list[Detection]] = {}

    counter = {"i": 0}

    def detector_fn(_frame):
        i = counter["i"]
        counter["i"] += 1
        return precomputed[i]

    cam = Retina(
        source_id="cam",
        detector=CallableDetector(detector_fn),
        tracker=IoUTracker(min_hits=1),
        rules=[
            ZoneRule(zone, classes={"person"}, dwell_s=1.0),
            LineRule(line, classes={"person"}),
            CountRule(threshold=1, classes={"person"}),
        ],
    )
    return cam, precomputed, counter


def _bench_detector_only(precomputed, n: int) -> float:
    """Time just the synthetic detector calls — the cost we subtract out."""
    counter = {"i": 0}

    def detector_fn(_frame):
        i = counter["i"]
        counter["i"] += 1
        return precomputed[i]

    t0 = time.perf_counter()
    for _ in range(n):
        detector_fn(None)
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=2000, help="frames to process")
    ap.add_argument("--tracks", type=int, default=20, help="objects per frame")
    ap.add_argument("--warmup", type=int, default=200, help="warmup frames (untimed)")
    args = ap.parse_args()

    w = h = 640
    n, k = args.frames, args.tracks

    cam, precomputed, counter = _build_retina(k, w, h)
    total = args.warmup + n
    for i in range(total):
        precomputed[i] = _make_detections(k, i, w, h)

    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Warmup (JIT-free Python, but warms caches / branch state).
    for i in range(args.warmup):
        cam.process(img, float(i))

    # Full pipeline (detector + tracker + rules + event build).
    t0 = time.perf_counter()
    for i in range(args.warmup, total):
        cam.process(img, float(i))
    full = time.perf_counter() - t0

    # Detector-only, same number of frames, to subtract it out.
    det = _bench_detector_only(precomputed, n)

    overhead_ms = (full - det) / n * 1e3
    full_ms = full / n * 1e3
    det_ms = det / n * 1e3

    print(f"frames={n}  tracks/frame={k}  (warmup={args.warmup})")
    print(f"  full pipeline : {full_ms:.4f} ms/frame")
    print(f"  detector stub : {det_ms:.4f} ms/frame  (excluded)")
    print(f"  Retina overhead: {overhead_ms:.4f} ms/frame at {k} tracks")


if __name__ == "__main__":
    main()
