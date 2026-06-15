"""AutoResearch: auto-tune a Retina pipeline from a few expensive oracle labels.

The story (this is a Retina selling point):
  1. A strong, expensive model (the "oracle" — here YOLO11l; swap for a frontier
     VLM / Opus on sampled clips) labels the events on the TRAIN segment.
  2. We search a cheap pipeline's params (tiny YOLO11n + tracker + rule) to
     reproduce those labels — maximizing event-F1.
  3. We check the tuned cheap pipeline GENERALIZES on a held-out TEST segment of
     the same scene (different time) — teacher hinted, student carries on.

Run (downloads YOLO weights on first use; caches detections to /tmp):
  pip install 'retina-sdk[yolo]' opencv-python
  python examples/autotune.py /tmp/demo.mp4
"""

import json
import os
import random
import sys

import numpy as np

from retina import IoUTracker, Line, LineRule, Retina
from retina.detect import Detection
from retina.eval import event_f1

FPS = 5
IMGSZ = 512
SPLIT_T = 32.0  # seconds: train [0, 32), test [32, 64)
VEHICLES = {"car", "truck", "bus", "motorcycle"}
# A vertical tripwire across the main road (pixel coords in the 1920x1080 frame;
# vehicles flow through the y≈560-680 band).
LINE = Line("road", (900.0, 540.0), (900.0, 700.0))
CACHE = "/tmp/retina_autotune_cache.json"
DUMMY = np.zeros((2, 2, 3), np.uint8)


def cache_detections(video: str) -> dict:
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            return json.load(f)
    import cv2
    from ultralytics import YOLO

    print("running detectors once (student=yolo11n, oracle=yolo11l)…", file=sys.stderr)
    student, oracle = YOLO("yolo11n.pt"), YOLO("yolo11l.pt")
    cap = cv2.VideoCapture(video)
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native / FPS))
    out = {"t": [], "student": [], "oracle": []}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            t = idx / native
            out["t"].append(t)
            for key, model, conf in (("student", student, 0.05), ("oracle", oracle, 0.4)):
                r = model.predict(frame, imgsz=IMGSZ, conf=conf, verbose=False)[0]
                dets = [
                    [r.names[int(b.cls)], *[float(v) for v in b.xyxy[0].tolist()], float(b.conf)]
                    for b in r.boxes
                ]
                out[key].append(dets)
        idx += 1
    cap.release()
    with open(CACHE, "w") as f:
        json.dump(out, f)
    return out


def to_dets(rows, conf, classes):
    return [
        Detection(label=r[0], bbox=(r[1], r[2], r[3], r[4]), confidence=r[5])
        for r in rows
        if r[5] >= conf and r[0] in classes
    ]


def run_pipeline(per_frame, ts, *, conf, iou, min_hits, stride, classes=VEHICLES):
    idxs = list(range(0, len(per_frame), stride))
    cached = [to_dets(per_frame[i], conf, classes) for i in idxs]

    class Cached:
        def __init__(self):
            self.k = 0

        def __call__(self, image):
            d = cached[self.k]
            self.k += 1
            return d

    cam = Retina(
        "cam",
        Cached(),
        tracker=IoUTracker(iou_threshold=iou, min_hits=min_hits),
        rules=[LineRule(LINE, classes=classes)],
    )
    return list(cam.run([(DUMMY, ts[i]) for i in idxs]))


def split(data):
    tr = [i for i, t in enumerate(data["t"]) if t < SPLIT_T]
    te = [i for i, t in enumerate(data["t"]) if t >= SPLIT_T]
    return tr, te


def main(video: str):
    data = cache_detections(video)
    tr, te = split(data)
    print(f"frames: {len(data['t'])} ({len(tr)} train / {len(te)} test) @ {FPS}fps")

    def slice_(key, idxs):
        return [data[key][i] for i in idxs], [data["t"][i] for i in idxs]

    s_tr, t_tr = slice_("student", tr)
    s_te, t_te = slice_("student", te)
    o_tr, _ = slice_("oracle", tr)
    o_te, _ = slice_("oracle", te)

    # Oracle reference events (strong model, fixed good config).
    ref_tr = run_pipeline(o_tr, t_tr, conf=0.4, iou=0.3, min_hits=2, stride=1)
    ref_te = run_pipeline(o_te, t_te, conf=0.4, iou=0.3, min_hits=2, stride=1)
    print(f"oracle line.cross events: {len(ref_tr)} train, {len(ref_te)} test")

    def score(student, ts, ref, p):
        pred = run_pipeline(student, ts, conf=p["conf"], iou=p["iou"], min_hits=p["min_hits"], stride=p["stride"])
        return event_f1(pred, ref)

    # Baseline: the obvious untuned cheap pipeline.
    base = {"conf": 0.25, "iou": 0.3, "min_hits": 3, "stride": 1}
    base_tr, base_te = score(s_tr, t_tr, ref_tr, base), score(s_te, t_te, ref_te, base)
    print(f"\nBASELINE (untuned)  {base}")
    print(f"  train F1={base_tr['f1']}   test F1={base_te['f1']}")

    # Search the cheap pipeline's params on TRAIN only, maximizing event-F1 vs oracle.
    rng = random.Random(0)
    best, best_f1 = base, base_tr["f1"]
    for _ in range(80):
        p = {
            "conf": round(rng.uniform(0.1, 0.6), 2),
            "iou": round(rng.uniform(0.2, 0.6), 2),
            "min_hits": rng.choice([1, 2, 3, 4]),
            "stride": rng.choice([1, 2, 3]),
        }
        f1 = score(s_tr, t_tr, ref_tr, p)["f1"]
        if f1 > best_f1:
            best, best_f1 = p, f1

    tuned_tr, tuned_te = score(s_tr, t_tr, ref_tr, best), score(s_te, t_te, ref_te, best)
    print(f"\nTUNED (autotuned)   {best}")
    print(f"  train F1={tuned_tr['f1']}   test F1={tuned_te['f1']}")
    print(
        "\nThe arbitrage: a tiny YOLO11n pipeline (~2.6M params), auto-tuned against a"
        f"\nYOLO11l oracle (~25M, ~10x bigger), reproduces its events at F1={tuned_te['f1']} on"
        "\nUNSEEN footage of the same scene. Swap the oracle for a frontier VLM/Opus on a"
        "\nfew sampled clips and the same loop builds a domain pipeline from sparse labels."
    )

    workflow = {
        "source_id": "cam_intersection",
        "nodes": [
            {"id": "det", "type": "yolo", "weights": "yolo11n.pt", "classes": sorted(VEHICLES), "min_confidence": best["conf"]},
            {"id": "trk", "type": "iou_tracker", "iou_threshold": best["iou"], "min_hits": best["min_hits"]},
            {"id": "road", "type": "line_rule", "a": list(LINE.a), "b": list(LINE.b), "normalized": False, "classes": sorted(VEHICLES)},
            {"id": "out", "type": "jsonl", "path": "events.jsonl"},
        ],
        "flow": ["det", "trk", "road", "out"],
    }
    with open("tuned_workflow.json", "w") as f:
        json.dump(workflow, f, indent=2)
    print("\nsaved tuned pipeline -> tuned_workflow.json (a ready-to-run Retina workflow)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/demo.mp4")
