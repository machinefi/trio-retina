"""One event stream, three applications — the *app-agnostic* demo.

The SAME Retina events drive a security app, a retail app, and a safety app.
Retina emits generic primitives (`line.cross`, `zone.dwell`, `count.threshold`);
meaning lives in the application, never in Retina. No model, no GPU.

    python examples/three_apps.py
"""

import numpy as np

from retina import CountRule, IoUTracker, Line, LineRule, Retina, Zone, ZoneRule
from retina.detect import Detection
from retina.events import EventType


class TwoWalkers:
    """Two people walk through a dock; person B follows one frame behind A."""

    def __init__(self):
        self.f = 0

    def __call__(self, frame):
        dets = []
        xa = self.f * 6
        if xa <= 96:
            dets.append(Detection("person", (xa - 10, 40, xa + 10, 60), 0.90))
        xb = (self.f - 1) * 6
        if 0 <= xb <= 96:
            dets.append(Detection("person", (xb - 10, 60, xb + 10, 80), 0.85))
        self.f += 1
        return dets


def build_events():
    dock = Zone("dock", [(40, 0), (60, 0), (60, 100), (40, 100)])
    door = Line("door", (50, 0), (50, 100))
    cam = Retina(
        "cam_dock",
        TwoWalkers(),
        tracker=IoUTracker(min_hits=2),
        rules=[
            ZoneRule(dock, classes={"person"}, dwell_s=2.0),
            LineRule(door, classes={"person"}),
            CountRule(threshold=2, classes={"person"}, zone=dock),
        ],
    )
    frames = [(np.zeros((100, 100, 3), np.uint8), float(i)) for i in range(22)]
    return list(cam.run(frames))


# --- three applications, each just a function over the same events ---


def security(ev):
    if ev.type == EventType.LINE_CROSS:
        return f"INTRUSION: {ev.label} #{ev.id} crossed the dock line (t={ev.t:.0f})"


def retail(ev):
    if ev.type == EventType.LINE_CROSS and ev.dir == "a_to_b":
        return f"footfall +1: {ev.label} entered (t={ev.t:.0f})"


def safety(ev):
    if ev.type == EventType.COUNT_THRESHOLD:
        return f"OVERCROWDING: {ev.n} people in dock (t={ev.t:.0f})"
    if ev.type == EventType.ZONE_DWELL:
        return f"extended dwell: #{ev.id} stayed {ev.dur:.0f}s in dock (t={ev.t:.0f})"


def main():
    events = build_events()
    print(f"Retina emitted {len(events)} generic events from one camera.\n")
    for name, app in [("SECURITY", security), ("RETAIL", retail), ("SAFETY", safety)]:
        print(f"── {name} app ──")
        for ev in events:
            line = app(ev)
            if line:
                print("  " + line)
        print()


if __name__ == "__main__":
    main()
