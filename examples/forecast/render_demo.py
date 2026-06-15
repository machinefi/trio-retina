"""Render an annotated demo video — boxes, track ids, trails, zone, events, and
the thing that sets Retina apart: a **forecast** arrow showing where each entity
is headed (0.6s ahead, from the dynamics on the WorldState).

    python examples/forecast/render_demo.py /tmp/demo.mp4 /tmp/retina_demo.mp4
"""

import colorsys
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dynamics import LinearForecaster  # noqa: E402

from retina import CountRule, IoUTracker, Line, LineRule, WorldState, YoloDetector, Zone, ZoneRule  # noqa: E402
from retina.nodes import DetectorNode, RuleNode, TrackerNode  # noqa: E402
from retina.pipeline import Pipeline  # noqa: E402
from retina.sources import video_frames  # noqa: E402

CLASSES = {"car", "truck", "bus", "motorcycle", "person"}
FPS, MAX_FRAMES, OUT_W = 10, 120, 1280
ROAD = [(0.10, 0.50), (0.95, 0.50), (0.95, 0.70), (0.10, 0.70)]
LINE = [(0.45, 0.46), (0.45, 0.72)]


def color(i):
    r, g, b = colorsys.hsv_to_rgb((i * 0.37) % 1.0, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def main(path, out):
    cap = cv2.VideoCapture(path)
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    stride = max(1, round(native / FPS))

    road = Zone("road", ROAD, normalized=True)
    pipe = Pipeline(
        [DetectorNode(YoloDetector("yolo11n.pt", classes=CLASSES, min_confidence=0.3)),
         TrackerNode(IoUTracker(min_hits=3, max_missed=8)),
         RuleNode(ZoneRule(road, classes=CLASSES, dwell_s=2.0)),
         RuleNode(LineRule(Line("mid", LINE[0], LINE[1], normalized=True), classes=CLASSES)),
         RuleNode(CountRule(threshold=5, classes=CLASSES, zone=road))],
        source_id="cam",
    )
    fc = LinearForecaster()
    trails: dict[int, list] = {}
    n_events = 0
    writer = None
    print(f"rendering {os.path.basename(path)} @ {FPS}fps…", file=sys.stderr)

    for img, t in video_frames(path, stride=stride, max_frames=MAX_FRAMES):
        f = pipe.process(img, t)
        fc.observe(WorldState.from_frame(f))
        n_events += len(f.events)
        H0, W0 = img.shape[:2]
        s = OUT_W / W0
        vis = cv2.resize(img, (OUT_W, int(H0 * s)))
        Hs, Ws = vis.shape[:2]

        # zone (translucent fill + outline) and tripwire
        overlay = vis.copy()
        poly = np.array([[int(x * Ws), int(y * Hs)] for x, y in ROAD], np.int32)
        cv2.fillPoly(overlay, [poly], (60, 180, 75))
        cv2.addWeighted(overlay, 0.18, vis, 0.82, 0, vis)
        cv2.polylines(vis, [poly], True, (60, 180, 75), 2)
        cv2.line(vis, (int(LINE[0][0] * Ws), int(LINE[0][1] * Hs)),
                 (int(LINE[1][0] * Ws), int(LINE[1][1] * Hs)), (0, 215, 255), 2)

        # entities, drawn DIM so the forecast is the star
        for trk in f.tracks:
            c = color(trk.track_id)
            x1, y1, x2, y2 = (int(v * s) for v in trk.bbox)
            cv2.rectangle(vis, (x1, y1), (x2, y2), c, 1)
            cv2.putText(vis, f"{trk.label} #{trk.track_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            tr = trails.setdefault(trk.track_id, [])
            tr.append((cx, cy))
            del tr[:-10]
            for a, b in zip(tr, tr[1:], strict=False):
                cv2.line(vis, a, b, c, 1, cv2.LINE_AA)

        # FORECAST (the star): translucent ghost box + thick magenta arrow, 1.0s ahead
        magenta = (255, 0, 255)
        pred_by_id = {e.id: e for e in fc.predict(1.0).entities}
        ghost = vis.copy()
        arrows = []
        for trk in f.tracks:
            pe = pred_by_id.get(str(trk.track_id))
            if pe is None:
                continue
            cx, cy = int((trk.bbox[0] + trk.bbox[2]) / 2 * s), int((trk.bbox[1] + trk.bbox[3]) / 2 * s)
            px1, py1, px2, py2 = (int(v * s) for v in pe.bbox)
            pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
            if (pcx - cx) ** 2 + (pcy - cy) ** 2 < 18 ** 2:  # essentially still
                continue
            cv2.rectangle(ghost, (px1, py1), (px2, py2), magenta, -1)
            arrows.append(((cx, cy), (pcx, pcy), (px1, py1, px2, py2)))
        cv2.addWeighted(ghost, 0.30, vis, 0.70, 0, vis)
        for a, b, box in arrows:
            cv2.rectangle(vis, box[:2], box[2:], magenta, 2)
            cv2.arrowedLine(vis, a, b, magenta, 4, cv2.LINE_AA, tipLength=0.28)

        # HUD
        cv2.rectangle(vis, (0, 0), (Ws, 34), (20, 20, 20), -1)
        cv2.putText(vis, "Retina  |  live world-state  +  1.0s FORECAST  (magenta = where it's going)",
                    (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, f"events: {n_events}", (Ws - 150, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        if writer is None:
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (Ws, Hs))
        writer.write(vis)

    if writer:
        writer.release()
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/demo.mp4",
         sys.argv[2] if len(sys.argv) > 2 else "/tmp/retina_demo.mp4")
