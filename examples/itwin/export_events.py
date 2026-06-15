"""Export a Retina event/state stream for the iTwin.js digital-twin demo.

Runs the SAME Retina traffic pipeline as the forecast demo (detect | track |
zone/line/count rules) plus a forecaster, and writes a single replayable
``retina_events.json``. The iTwin viewer (TypeScript) consumes only this file —
it never touches Python, a model, or pixels. That seam is the whole point:

    camera ─▶ Retina (any backbone) ─▶ retina.event + WorldState ─▶ iTwin twin

Each entity carries BOTH its image-space centroid and a **world** ground-plane
coordinate, obtained by a one-time camera→world homography (4 reference points —
exactly the per-camera calibration you would do in production). The iTwin
``RetinaDecorator`` drops a marker at the world point on the iModel ground plane,
shows the live events as tooltips/alerts, and draws the forecast arrow.

    python examples/itwin/export_events.py /tmp/demo.mp4 examples/itwin/retina_events.json
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "forecast"))
from dynamics import LearnedForecaster, LinearForecaster  # noqa: E402

from retina import (  # noqa: E402
    CountRule,
    IoUTracker,
    Line,
    LineRule,
    WorldState,
    YoloDetector,
    Zone,
    ZoneRule,
)
from retina.nodes import DetectorNode, RuleNode, TrackerNode  # noqa: E402
from retina.pipeline import Pipeline  # noqa: E402
from retina.sources import video_frames  # noqa: E402

CLASSES = {"car", "truck", "bus", "motorcycle", "person"}
FPS, MAX_FRAMES = 5, 90
ROAD = [(0.10, 0.50), (0.95, 0.50), (0.95, 0.70), (0.10, 0.70)]  # normalized image quad
LINE = [(0.45, 0.46), (0.45, 0.72)]
# The road quad above maps to this metric rectangle on the iModel ground plane.
# (one-time per-camera calibration — 4 correspondences is all a homography needs.)
WORLD_RECT = [(0.0, 0.0), (60.0, 0.0), (60.0, 12.0), (0.0, 12.0)]  # metres


def homography(img_w: int, img_h: int) -> np.ndarray:
    src = np.array([[x * img_w, y * img_h] for x, y in ROAD], dtype=np.float32)
    dst = np.array(WORLD_RECT, dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def to_world(H: np.ndarray, cx: float, cy: float) -> list[float]:
    p = cv2.perspectiveTransform(np.array([[[cx, cy]]], dtype=np.float32), H)[0][0]
    return [round(float(p[0]), 2), round(float(p[1]), 2)]


def in_poly(cx: float, cy: float, poly_px) -> bool:
    return cv2.pointPolygonTest(np.array(poly_px, dtype=np.int32), (cx, cy), False) >= 0


def main(path: str, out: str) -> None:
    road = Zone("road", ROAD, normalized=True)
    pipe = Pipeline(
        [
            DetectorNode(YoloDetector("yolo11n.pt", classes=CLASSES, min_confidence=0.3)),
            TrackerNode(IoUTracker(min_hits=3, max_missed=8)),
            RuleNode(ZoneRule(road, classes=CLASSES, dwell_s=2.0)),
            RuleNode(LineRule(Line("mid", LINE[0], LINE[1], normalized=True), classes=CLASSES)),
            RuleNode(CountRule(threshold=5, classes=CLASSES, zone=road)),
        ],
        source_id="cam:intersection-01",
    )
    ckpt = os.path.join(os.path.dirname(__file__), "..", "forecast", "dynamics.ckpt")
    forecaster = LearnedForecaster(ckpt) if os.path.exists(ckpt) else LinearForecaster()
    fc_name = type(forecaster).__name__

    cap = cv2.VideoCapture(path)
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    stride = max(1, round(native / FPS))

    frames_out: list[dict] = []
    img_w = img_h = 0
    H = None
    n_events = 0

    for img, t in video_frames(path, stride=stride, max_frames=MAX_FRAMES):
        if H is None:
            img_h, img_w = img.shape[:2]
            H = homography(img_w, img_h)
            road_px = [(x * img_w, y * img_h) for x, y in ROAD]
        f = pipe.process(img, t)
        ws = WorldState.from_frame(f)
        forecaster.observe(ws)
        pred = {e.id: e for e in forecaster.predict(1.0).entities}

        ents = []
        for trk in f.tracks:
            x1, y1, x2, y2 = trk.bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ent = {
                "id": str(trk.track_id),
                "type": trk.label,
                "img": [round(cx, 1), round(cy, 1)],
                "world": to_world(H, cx, cy),
                "zone": "road" if in_poly(cx, cy, road_px) else None,
            }
            pe = pred.get(str(trk.track_id))
            if pe is not None and pe.bbox is not None:
                px, py = (pe.bbox[0] + pe.bbox[2]) / 2, (pe.bbox[1] + pe.bbox[3]) / 2
                ent["forecast"] = {"world": to_world(H, px, py), "horizon_s": 1.0}
            if ent["zone"] is None:
                del ent["zone"]
            ents.append(ent)

        events = [e.to_dict() for e in f.events]
        n_events += len(events)
        frames_out.append({"t": round(t, 3), "entities": ents, "events": events})

    doc = {
        "meta": {
            "schema": "retina.itwin/0.1",
            "source": "cam:intersection-01",
            "fps": FPS,
            "image_size": [img_w, img_h],
            "forecaster": fc_name,
            "world": {
                "units": "m",
                "ground_plane": "z=0",
                "road_rect_m": WORLD_RECT,
                "note": "entity.world = camera->world homography of the image centroid (4-point calibration)",
            },
            "homography_img_to_world": H.tolist() if H is not None else None,
        },
        "frames": frames_out,
    }
    with open(out, "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    print(
        f"wrote {out}  ·  {len(frames_out)} frames @ {FPS}fps  ·  "
        f"{n_events} events  ·  forecaster={fc_name}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "/tmp/demo.mp4",
        sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "retina_events.json"),
    )
