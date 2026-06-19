"""Record a Retina WorldState sequence for the PASS-MAP PoC from a real clip.

Honest data layer for the pass-map demo. Nothing synthetic: a real broadcast-ish
clip (`sports.mp4`, a short roboflow `sports` sample) goes through a real generic
COCO YOLO detector, then the Retina interop path, and comes out as a standardized
sequence of per-frame player tracks + ball detections that the pass detector reads.

Pipeline (Retina is the standardized state layer in the middle):

    frame (cv2, BGR)
      -> YOLO detect persons + sports ball   (ultralytics, generic COCO yolo11*)
      -> sv.Detections                        (supervision — the interop format)
      -> Detection.from_supervision           (Retina's documented interop path)
      -> DetectorNode | TrackerNode           (Retina pipeline: IoU tracking, ids)
      -> WorldState.from_frame                 (the standardized snapshot)

Players (`person`) get stable track ids through the Retina tracker; the BALL is
NOT tracked as a Retina entity (generic COCO is too flaky on the small fast ball
to make a stable track) — it is kept as a separate per-frame signal: the single
highest-confidence `sports ball` detection in the frame, or null.

Teams are assigned by jersey colour: mean torso-region HSV-hue per player crop,
k-means(k=2). Goalkeeper / referee (pink, lime-vest) usually fall out as outliers;
we keep them as their nearest cluster but flag low-confidence colour cases.

Run (from repo root), Mac Studio MPS:
    python examples/world_model/soccer/passmap/record_passmap.py \
        --clip /Users/rc/Desktop/2026/playground-assets/sports.mp4 \
        --weights yolo11m.pt \
        --out examples/world_model/soccer/passmap/data/passmap_states.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

# Allow `python examples/world_model/soccer/passmap/record_passmap.py` from root.
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ),
)

from retina import IoUTracker, WorldState  # noqa: E402
from retina.detect import Detection  # noqa: E402
from retina.nodes import DetectorNode, TrackerNode  # noqa: E402
from retina.pipeline import Pipeline  # noqa: E402

# COCO class ids: 0 = person, 32 = sports ball.
PERSON_CLS = 0
BALL_CLS = 32


def torso_color(frame_bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> tuple[float, ...]:
    """Mean HSV of the upper-centre (torso) region of a player bbox.

    Returns (H, S, V) in OpenCV ranges (H 0-179). Falls back to the whole box
    if the torso slice is empty."""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)
    # torso: vertical 15%..55% of the box, horizontal central 60%.
    ty1 = y1 + int(0.15 * h)
    ty2 = y1 + int(0.55 * h)
    tx1 = x1 + int(0.20 * w)
    tx2 = x2 - int(0.20 * w)
    H, W = frame_bgr.shape[:2]
    ty1, ty2 = max(0, ty1), min(H, ty2)
    tx1, tx2 = max(0, tx1), min(W, tx2)
    if ty2 <= ty1 or tx2 <= tx1:
        crop = frame_bgr[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
    else:
        crop = frame_bgr[ty1:ty2, tx1:tx2]
    if crop.size == 0:
        return (0.0, 0.0, 0.0)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return tuple(float(v) for v in hsv.reshape(-1, 3).mean(0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--weights", default="yolo11m.pt")
    ap.add_argument("--out", default="examples/world_model/soccer/passmap/data/passmap_states.json")
    ap.add_argument("--frames-out", default="examples/world_model/soccer/passmap/data/frames")
    ap.add_argument("--step", type=int, default=1, help="sample every Nth video frame")
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf-person", type=float, default=0.35)
    ap.add_argument("--conf-ball", type=float, default=0.10, help="low: COCO ball recall is weak")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dump-frames", action="store_true", help="also write sampled jpgs")
    args = ap.parse_args()

    import supervision as sv
    from ultralytics import YOLO

    model = YOLO(args.weights)

    class _Holder:
        dets: list[Detection] = []

        def __call__(self, _image):
            return self.dets

    holder = _Holder()
    # Retina pipeline: detector holder + IoU tracker. min_hits=1 so a player is
    # tracked the frame it appears; this clip pans hard so we keep a forgiving
    # max_missed and a moderate IoU threshold.
    pipe = Pipeline(
        [
            DetectorNode(holder),
            TrackerNode(IoUTracker(min_hits=1, iou_threshold=0.2, max_missed=6)),
        ],
        source_id="sports",
    )

    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f"clip {os.path.basename(args.clip)}: {W}x{H} @ {fps:.1f}fps, {n_total} frames "
        f"({n_total / fps:.1f}s); sampling every {args.step}, max {args.max_frames}",
        flush=True,
    )

    if args.dump_frames:
        os.makedirs(args.frames_out, exist_ok=True)

    states: list[dict] = []
    ball_hits = 0
    fi = -1
    while len(states) < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if fi % args.step:
            continue
        t = round(fi / fps, 3)

        res = model(
            frame,
            imgsz=args.imgsz,
            conf=min(args.conf_person, args.conf_ball),
            classes=[PERSON_CLS, BALL_CLS],
            verbose=False,
        )[0]
        sv_all = sv.Detections.from_ultralytics(res)

        # split persons (-> Retina tracking) and ball (separate per-frame signal)
        cls_ids = sv_all.class_id if sv_all.class_id is not None else np.array([])
        person_mask = np.array(
            [c == PERSON_CLS and sv_all.confidence[i] >= args.conf_person for i, c in enumerate(cls_ids)]
        )
        ball_mask = np.array(
            [c == BALL_CLS and sv_all.confidence[i] >= args.conf_ball for i, c in enumerate(cls_ids)]
        )

        sv_person = sv_all[person_mask] if len(cls_ids) else sv_all
        holder.dets = Detection.from_supervision(sv_person)
        rframe = pipe.process(frame, t)  # detect + track via Retina
        ws = WorldState.from_frame(rframe)  # standardized snapshot

        ents = []
        for e in ws.entities:
            if e.bbox is None:
                continue
            x1, y1, x2, y2 = e.bbox
            fx = (x1 + x2) / 2.0  # foot point: bottom-centre
            fy = y2
            hsv = torso_color(frame, e.bbox)
            ents.append(
                {
                    "id": e.id,
                    "cx": round(float(fx), 2),
                    "cy": round(float(fy), 2),
                    "bbox": [round(float(b), 1) for b in (x1, y1, x2, y2)],
                    "hsv": [round(v, 2) for v in hsv],
                    "conf": round(float(e.conf or 0.0), 3),
                }
            )

        # ball: single highest-confidence detection in the frame (or null)
        ball = None
        if len(cls_ids) and ball_mask.any():
            bxy = sv_all.xyxy[ball_mask]
            bcf = sv_all.confidence[ball_mask]
            j = int(np.argmax(bcf))
            bx1, by1, bx2, by2 = (float(v) for v in bxy[j])
            ball = {
                "cx": round((bx1 + bx2) / 2.0, 2),
                "cy": round((by1 + by2) / 2.0, 2),
                "conf": round(float(bcf[j]), 3),
            }
            ball_hits += 1

        states.append({"t": t, "frame_idx": fi, "entities": ents, "ball": ball})
        if args.dump_frames:
            cv2.imwrite(os.path.join(args.frames_out, f"{len(states) - 1:04d}.jpg"), frame)
        if len(states) % 25 == 0:
            print(
                f"  {len(states)} frames (t={t:.2f}s, {len(ents)} players, "
                f"ball_hits={ball_hits})",
                flush=True,
            )

    cap.release()

    n_players = len({e["id"] for st in states for e in st["entities"]})
    out = {
        "spec": "retina.world_model.soccer.passmap/0.1",
        "scene": "real soccer clip; players detected+tracked via Retina, foot-point "
        "position; ball = per-frame highest-conf COCO 'sports ball' detection",
        "clip": os.path.basename(args.clip),
        "weights": args.weights,
        "W": W,
        "H": H,
        "fps": fps,
        "step": args.step,
        "n_frames": len(states),
        "ball_hits": ball_hits,
        "n_player_tracks": n_players,
        "states": states,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(out, fp, separators=(",", ":"))
    size_kb = os.path.getsize(args.out) / 1024
    rate = ball_hits / max(1, len(states))
    print(
        f"wrote {args.out} ({size_kb:.0f} KB): {len(states)} frames, "
        f"{n_players} player tracks, ball-detection rate {rate:.1%} "
        f"({ball_hits}/{len(states)})",
        flush=True,
    )


if __name__ == "__main__":
    main()
