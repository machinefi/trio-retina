"""Forecast on REAL video — the baseline number a learned dynamics model must beat.

Retina (YOLO + tracker) turns a real fixed-cam clip into a `WorldState` stream;
the constant-velocity baseline predicts each entity's position `H` frames ahead;
we score against the actual future state Retina computed from the future frames.
On real motion (cars turning / accelerating) the velocity model beats no-motion
but is far from perfect — that gap is what a learned, interaction-aware dynamics
model is for.

    pip install 'trio-retina[yolo,video]'
    python examples/forecast/forecast_video.py /tmp/demo.mp4
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(__file__))
from dynamics import LinearForecaster, forecast_error  # noqa: E402

from retina import IoUTracker, WorldState, YoloDetector  # noqa: E402
from retina.nodes import DetectorNode, TrackerNode  # noqa: E402
from retina.pipeline import Pipeline  # noqa: E402
from retina.sources import video_frames  # noqa: E402

CLASSES = {"car", "truck", "bus", "motorcycle", "person"}
FPS = 5
MAX_FRAMES = 180
H_FRAMES = 3  # predict 3 sampled frames ahead


def main(path):
    cap = cv2.VideoCapture(path)
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    stride = max(1, round(native / FPS))
    horizon_s = H_FRAMES / FPS

    pipe = Pipeline(
        [DetectorNode(YoloDetector("yolo11n.pt", classes=CLASSES, min_confidence=0.3)),
         TrackerNode(IoUTracker(min_hits=3, max_missed=8))],
        source_id="cam",
    )
    print(f"running Retina (yolo11n) on {os.path.basename(path)} @ {FPS}fps…", file=sys.stderr)
    gt = [WorldState.from_frame(pipe.process(img, t))
          for img, t in video_frames(path, stride=stride, max_frames=MAX_FRAMES)]

    fc = LinearForecaster()
    lin, naive = [], []
    for ti, state in enumerate(gt):
        fc.observe(state)
        tgt = ti + H_FRAMES
        if tgt >= len(gt):
            break
        actual = gt[tgt]
        el = forecast_error(fc.predict(horizon_s), actual)
        en = forecast_error(gt[ti], actual)  # no-motion
        if el["mae"] is not None:
            lin.append(el["mae"])
        if en["mae"] is not None:
            naive.append(en["mae"])

    def mean(xs):
        return round(sum(xs) / len(xs), 1) if xs else None

    n_ent = len({e.id for s in gt for e in s.entities})
    print(f"\nframes: {len(gt)}   entities tracked: {n_ent}   "
          f"horizon: {H_FRAMES} frames ({horizon_s:.1f}s)\n")
    print("forecast error (mean centroid px, real motion):")
    print(f"  no-motion baseline:      {mean(naive)}")
    print(f"  constant-velocity (L4):  {mean(lin)}")
    if lin and naive:
        cut = round(100 * (1 - mean(lin) / mean(naive)))
        print(f"\nvelocity baseline cuts the error {cut}% vs no-motion — a real number on real\n"
              "motion. The residual (turns, accel, interactions) is what a learned model targets.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/demo.mp4")
