"""Plug an existing Roboflow Supervision pipeline into Retina's event layer.

Runs with NO model and NO GPU: it builds tiny FAKE `sv.Detections`-like objects
inline (a SimpleNamespace with `.xyxy / .confidence / .class_id / .data`) and
feeds them through `Detection.from_supervision(...)` -> `ZoneRule` to print
tracked `zone.enter` / `zone.dwell` events.

    python examples/from_supervision.py

In real use you delete the fake and pass your actual `sv.Detections` straight
in — Retina never imports `supervision`, it just duck-types the object:

    from retina.detect import Detection
    dets = Detection.from_supervision(sv_detections)   # your real detections
"""

from types import SimpleNamespace

import numpy as np

from retina import IoUTracker, Retina, Zone, ZoneRule
from retina.detect import Detection


def fake_sv_detections(x: float):
    """A stand-in for `sv.Detections`: one 'person' box at horizontal pos `x`.

    Mirrors Supervision's attribute shape exactly — `.xyxy` (Nx4 array),
    `.confidence` / `.class_id` (length-N arrays), `.data` (dict, may carry
    `class_name`). Your real `sv.Detections` already has all of these.
    """
    return SimpleNamespace(
        xyxy=np.array([[x - 10, 40, x + 10, 60]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
        data={"class_name": np.array(["person"])},
    )


class SupervisionAdapter:
    """A detector that turns each frame's `sv.Detections` into `Detection`s.

    Here we synthesize the Supervision object per frame; in production this is
    where you'd run your existing Supervision pipeline and pass its output to
    `Detection.from_supervision(...)`."""

    def __init__(self):
        self.f = 0

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        x = min(self.f * 6, 50)  # walk in, then linger in the zone to dwell
        self.f += 1
        sv_dets = fake_sv_detections(x)
        return Detection.from_supervision(sv_dets)


def main() -> None:
    dock = Zone("dock", [(0.4, 0), (0.6, 0), (0.6, 1), (0.4, 1)], normalized=True)

    cam = Retina(
        source_id="cam_01",
        detector=SupervisionAdapter(),
        tracker=IoUTracker(min_hits=2),
        rules=[ZoneRule(dock, classes={"person"}, dwell_s=3.0)],
    )

    frames = [(np.zeros((100, 100, 3), np.uint8), float(i)) for i in range(18)]
    for event in cam.run(frames):
        print(event.to_json())


if __name__ == "__main__":
    main()
