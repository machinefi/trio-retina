"""Cheap gate, fewer model calls — the *efficiency* demo.

A motion gate skips the detector on static frames. Same events, far fewer
detector invocations. This is the exact seam where you'd later gate an expensive
VLM behind a cheap always-on signal (the academic cascade pattern). End-to-end
pixels -> events, pure numpy, no model.

    python examples/gate_savings.py
"""

import numpy as np

from retina import IoUTracker, MotionGate, Retina, Zone, ZoneRule


def make_frames():
    """100 frames; a white square crosses the scene only during frames 40..60."""
    frames = []
    for f in range(100):
        img = np.zeros((80, 120, 3), np.uint8)
        if 40 <= f <= 60:
            x = 10 + (f - 40) * 5
            img[30:50, x : x + 16] = 255
        frames.append((img, float(f)))
    return frames


class BlobDetector:
    """A real (tiny) detector: bounding box of the bright pixels. Counts calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, frame):
        self.calls += 1
        mask = frame.any(axis=2)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return []
        from retina.detect import Detection

        return [Detection("person", (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())), 0.95)]


def run(gate):
    det = BlobDetector()
    dock = Zone("dock", [(0.3, 0), (0.7, 0), (0.7, 1), (0.3, 1)], normalized=True)
    cam = Retina("cam", det, tracker=IoUTracker(min_hits=1), rules=[ZoneRule(dock)], gate=gate)
    events = [e.type for e in cam.run(make_frames())]
    return det.calls, events


def main():
    calls_off, events_off = run(gate=None)
    calls_on, events_on = run(gate=MotionGate())
    saved = 100 * (1 - calls_on / calls_off)
    print(f"detector calls without gate: {calls_off}")
    print(f"detector calls with gate:    {calls_on}   ({saved:.0f}% fewer)")
    print(f"events identical: {events_off == events_on}  ({events_off})")


if __name__ == "__main__":
    main()
