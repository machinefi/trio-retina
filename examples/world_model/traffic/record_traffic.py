"""Real footage → Retina WorldState + speed, the honest pipeline.

Nothing synthetic here: a traffic clip goes through a real YOLO vehicle detector,
a real tracker, Retina's standardized `WorldState`, a hand-calibrated ground
homography (pixels → metres on `Entity.locus`), and the same `SpeedEstimator`
the synthetic demo uses. Out comes a JSON of per-frame states + `speed` events —
a from-scratch "speed radar", assembled from Retina primitives.

    frame (cv2, BGR)
      -> YOLO vehicle detection            (ultralytics; car/truck/bus/motorcycle)
      -> sv.Detections -> Detection.from_supervision   (Supervision interop)
      -> DetectorNode | TrackerNode        (Retina pipeline: stable track ids)
      -> WorldState.from_frame             (standardized snapshot)
      -> calib.to_metres(bbox foot)        (-> Entity.locus, metres)
      -> SpeedEstimator                    (d locus / dt -> km/h + trap events)

Calibration is a small JSON — four image points and the real-world metres they
correspond to on the road plane (lane widths, dashed-segment lengths, a surveyed
box). See `calib.example.json`. This is the one manual step, and it is *the*
step: get it right and every pixel is metres.

    python record_traffic.py --clip road.mp4 --weights yolo11l.pt \
        --calib calib.example.json --out states.json

Heavy deps (cv2, ultralytics, torch) are imported lazily; the numpy-only core
(homography / speed / synthetic) never touches them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from retina import IoUTracker, WorldState  # noqa: E402
from retina.detect import Detection  # noqa: E402
from retina.events import Frame  # noqa: E402

from homography import RoadCalibration  # noqa: E402
from speed import SpeedEstimator  # noqa: E402

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


def load_calibration(path: str) -> tuple[RoadCalibration, float]:
    with open(path) as f:
        c = json.load(f)
    calib = RoadCalibration.from_correspondences(c["image_pts"], c["world_pts_m"])
    return calib, float(c.get("trap_x", 30.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--weights", default="yolo11l.pt")
    ap.add_argument("--calib", required=True, help="JSON: image_pts, world_pts_m, trap_x")
    ap.add_argument("--out", default="states.json")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    calib, trap_x = load_calibration(args.calib)
    model = YOLO(args.weights)
    names = model.names
    tracker = IoUTracker(min_hits=2)
    estimator = SpeedEstimator(src=os.path.basename(args.clip), trap_x=trap_x)

    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames_out = []
    events_out = []
    i = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % args.stride:
            continue
        t = i / fps
        res = model.predict(frame, conf=args.conf, verbose=False)[0]

        # ultralytics -> Retina Detections (via the documented Supervision path
        # when available; here we map directly to keep the dep surface minimal).
        dets = []
        for b in res.boxes:
            label = names[int(b.cls)]
            if label not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            dets.append(Detection(label=label, bbox=(x1, y1, x2, y2), confidence=float(b.conf)))

        fr = Frame(frame_num=i, src=estimator.src, t=t, detections=dets,
                   width=frame.shape[1], height=frame.shape[0])
        fr.tracks = tracker.update(dets, t=t)
        ws = WorldState.from_frame(fr)
        for ent in ws.entities:
            ent.locus = calib.to_metres(ent.bbox)   # pixels -> metres

        events_out.extend(e.to_dict() for e in estimator.update(t, ws))
        frames_out.append({
            "frame": i,
            "t": round(t, 3),
            "entities": [
                {"id": e.id, "type": e.type, "bbox": [round(c, 1) for c in e.bbox],
                 "locus_m": [round(c, 2) for c in e.locus],
                 "speed_kmh": e.attrs.get("speed_kmh")}
                for e in ws.entities
            ],
        })
    cap.release()

    with open(args.out, "w") as f:
        json.dump({"fps": fps, "trap_x": trap_x, "frames": frames_out, "events": events_out}, f)
    print(f"wrote {args.out}: {len(frames_out)} frames, {len(events_out)} speed events")
    for ev in events_out:
        print("  ", json.dumps(ev, separators=(",", ":")))


if __name__ == "__main__":
    main()
