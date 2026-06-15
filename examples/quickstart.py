"""Quickstart — runs with NO model and NO GPU.

Demonstrates the full Signal -> Event path with a scripted detector (a single
"person" walking left-to-right across a dock zone). Run it:

    python examples/quickstart.py

For a real model, swap `ScriptedDetector` for `YoloDetector("yolo11n.pt")` and
`frames` for `video_frames("dock.mp4")`.
"""

import numpy as np

from retina import CountRule, IoUTracker, Line, LineRule, Retina, Zone, ZoneRule
from retina.detect import Detection


class ScriptedDetector:
    """Returns one 'person' box marching across the frame, one step per call."""

    def __init__(self):
        self._xs = list(range(0, 102, 6))
        self._i = 0

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        if self._i >= len(self._xs):
            return []
        x = self._xs[self._i]
        self._i += 1
        return [Detection(label="person", bbox=(x - 10, 40, x + 10, 60), confidence=0.9)]


def main() -> None:
    dock = Zone("dock", [(40, 0), (60, 0), (60, 100), (40, 100)])
    tripwire = Line("door", (50, 0), (50, 100))

    cam = Retina(
        source_id="cam_01",
        detector=ScriptedDetector(),
        tracker=IoUTracker(min_hits=2),
        rules=[
            ZoneRule(dock, classes={"person"}, dwell_s=2.0),
            LineRule(tripwire, classes={"person"}),
            CountRule(threshold=1, classes={"person"}),
        ],
    )

    frames = [(np.zeros((100, 100, 3), dtype=np.uint8), float(i)) for i in range(18)]
    for event in cam.run(frames):
        print(event.to_json())


if __name__ == "__main__":
    main()
