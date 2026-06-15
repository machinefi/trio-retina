"""Swap the model, keep everything else — the *model-agnostic* demo.

The zone, rules, tracker, events, and sinks are identical. Only the detector
changes. A detector is just `callable(frame) -> list[Detection]`, so a scripted
stub, a plain function, a YOLO model, or a VLM behind an HTTP call all plug into
the same seam.

    python examples/any_model.py
"""

import numpy as np

from retina import CallableDetector, IoUTracker, Retina, Zone, ZoneRule
from retina.detect import Detection


def make_cam(detector):
    # Identical for every model — this is the whole point of a middle layer.
    dock = Zone("dock", [(0.4, 0), (0.6, 0), (0.6, 1), (0.4, 1)], normalized=True)
    return Retina(
        "cam",
        detector,
        tracker=IoUTracker(min_hits=2),
        rules=[ZoneRule(dock, classes={"person"}, dwell_s=2.0)],
    )


FRAMES = [(np.zeros((100, 100, 3), np.uint8), float(i)) for i in range(18)]


class ScriptedDetector:
    """A stub 'model' that walks one person across the frame."""

    def __init__(self):
        self.f = 0

    def __call__(self, frame):
        x = self.f * 6
        self.f += 1
        return [Detection("person", (x - 10, 40, x + 10, 60), 0.9)] if x <= 96 else []


_stub = ScriptedDetector()  # one stateful stub so the walker advances per frame


def plain_function_model(frame):
    """Any function works — pretend this ran YOLO / RT-DETR / a VLM and parsed
    the result into Detections. Returned shape is all Retina needs."""
    return _stub(frame)


# Real models look exactly the same — just construct a different detector:
#
#   from retina import YoloDetector
#   detector = YoloDetector("yolo11n.pt", classes={"person"})
#
#   # VLM-as-detector: ask a frontier model, parse boxes into Detections
#   def vlm_detector(frame):
#       boxes = my_vlm.detect(frame, prompt="people")   # your call
#       return [Detection("person", b.xyxy, b.score) for b in boxes]


def run(label, detector):
    print(f"── detector: {label} ──")
    for ev in make_cam(detector).run(list(FRAMES)):
        print("  " + ev.to_json())
    print()


if __name__ == "__main__":
    run("ScriptedDetector (class)", ScriptedDetector())
    run("plain function via CallableDetector", CallableDetector(plain_function_model))
