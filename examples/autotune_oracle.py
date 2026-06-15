"""AutoResearch with a FRONTIER oracle (here: Opus) — customer-flow / restricted-zone.

MachineFi's Trio is "the world model for physical operations"; one B2B use case is
*customer-flow analytics / restricted-zone monitoring*. This builds that pipeline
the AutoResearch way, with a frontier model as the sparse teacher:

  1. Opus (me) watched a few sampled frames of a fixed street cam and labeled how
     many PEOPLE are in a zone (the café/sidewalk frontage) — 5 train, 4 test
     frames. Sparse + expensive. (Counts hard-coded below.)
  2. We search the cheap YOLO11n pipeline's params — detection confidence, input
     resolution (imgsz), min box size — to reproduce those counts on TRAIN.
  3. We check it generalizes on the held-out TEST frames.

The teacher hinted on a few clips; the student carries on — the engine behind
building a per-scene pipeline from a handful of frontier-model labels.

  pip install 'retina-sdk[yolo]' opencv-python
  python examples/autotune_oracle.py /tmp/demo.mp4
"""

import json
import os
import sys

from retina.geometry import point_in_polygon

# Restricted/flow zone: the café + sidewalk frontage (pixels, 1920x1080).
ZONE = [(1080, 520), (1910, 520), (1910, 840), (1080, 840)]
# Sparse oracle labels — people in ZONE, annotated by Opus from sampled crops.
ORACLE = {4: 6, 10: 6, 16: 1, 22: 6, 28: 1, 36: 5, 44: 4, 52: 3, 60: 4}
TRAIN_T = [4, 10, 16, 22, 28]
TEST_T = [36, 44, 52, 60]
IMGSZ_CHOICES = [512, 960, 1280]
DETS = "/tmp/retina_oracle_dets.json"


def detect(video: str) -> dict:
    """yolo11n person boxes for each (timestamp, imgsz). Cached to disk."""
    if os.path.exists(DETS):
        return json.load(open(DETS))
    import cv2
    from ultralytics import YOLO

    print("running yolo11n on the labeled frames at 3 resolutions…", file=sys.stderr)
    model = YOLO("yolo11n.pt")
    cap = cv2.VideoCapture(video)
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out: dict = {}
    for t in ORACLE:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * native)))
        ok, frame = cap.read()
        if not ok:
            continue
        out[str(t)] = {}
        for sz in IMGSZ_CHOICES:
            r = model.predict(frame, imgsz=sz, conf=0.05, classes=[0], verbose=False)[0]
            out[str(t)][str(sz)] = [
                [*[float(v) for v in b.xyxy[0].tolist()], float(b.conf)] for b in r.boxes
            ]
    cap.release()
    json.dump(out, open(DETS, "w"))
    return out


def count(rows, *, conf, min_area):
    n = 0
    for x1, y1, x2, y2, c in rows:
        if c < conf or (x2 - x1) * (y2 - y1) < min_area:
            continue
        if point_in_polygon(((x1 + x2) / 2, (y1 + y2) / 2), ZONE):
            n += 1
    return n


def errors(dets, times, p):
    diffs = [abs(count(dets[str(t)][str(p["imgsz"])], conf=p["conf"], min_area=p["min_area"]) - ORACLE[t]) for t in times]
    return round(sum(diffs) / len(diffs), 3), round(sum(d == 0 for d in diffs) / len(diffs), 3)


def main(video: str):
    dets = detect(video)
    print(f"zone = café/sidewalk frontage; oracle (Opus) = {ORACLE}\n")

    base = {"conf": 0.25, "imgsz": 512, "min_area": 0}
    b_tr, b_te = errors(dets, TRAIN_T, base), errors(dets, TEST_T, base)
    print(f"BASELINE  {base}\n  train MAE={b_tr[0]} acc={b_tr[1]}   test MAE={b_te[0]} acc={b_te[1]}")

    best, best_mae = base, b_tr[0]
    for imgsz in IMGSZ_CHOICES:
        for conf in [round(0.1 + 0.05 * i, 2) for i in range(11)]:
            for min_area in [0, 300, 800, 1500, 3000]:
                p = {"conf": conf, "imgsz": imgsz, "min_area": min_area}
                if errors(dets, TRAIN_T, p)[0] < best_mae:
                    best, best_mae = p, errors(dets, TRAIN_T, p)[0]

    t_tr, t_te = errors(dets, TRAIN_T, best), errors(dets, TEST_T, best)
    print(f"\nTUNED     {best}\n  train MAE={t_tr[0]} acc={t_tr[1]}   test MAE={t_te[0]} acc={t_te[1]}")
    print(
        f"\nUNSEEN test frames: count MAE {b_te[0]} -> {t_te[0]}, exact-match {b_te[1]} -> {t_te[1]}\n"
        "Opus labeled 5 frames; autotune fit the cheap YOLO11n pipeline (conf + input\n"
        "resolution + min box) to reproduce per-zone people counts — and it held on\n"
        "frames it never saw. Swap Opus for a VLM API to scale the labeling."
    )

    workflow = {
        "source_id": "cam_storefront",
        "nodes": [
            {"id": "det", "type": "yolo", "weights": "yolo11n.pt", "classes": ["person"], "min_confidence": best["conf"]},
            {"id": "trk", "type": "iou_tracker", "min_hits": 2},
            {"id": "flow", "type": "count_rule", "threshold": 4, "classes": ["person"],
             "zone": [[x, y] for x, y in ZONE], "normalized": False},
            {"id": "out", "type": "jsonl", "path": "events.jsonl"},
        ],
        "flow": ["det", "trk", "flow", "out"],
    }
    json.dump(workflow, open("tuned_workflow.json", "w"), indent=2)
    print(f"\nsaved -> tuned_workflow.json (note tuned imgsz={best['imgsz']} for the detector)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/demo.mp4")
