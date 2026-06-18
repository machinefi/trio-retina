"""Record Retina WorldState sequences from REAL soccer footage.

This is the flagship world-model demo's data layer — the honest one. Nothing is
synthetic: a short clip of real broadcast match footage goes through a real
player detector, a real tracker, a real frozen DINOv2 appearance encoder, and
comes out the other side as standardized Retina `WorldState` sequences that a
learned dynamics model can train on.

Pipeline (Retina is the standardized state layer in the middle):

    frame (cv2, BGR)
      -> YOLO person detection            (ultralytics)
      -> sv.Detections                    (supervision — the de-facto interop)
      -> Detection.from_supervision        (Retina's documented interop path)
      -> DetectorNode | TrackerNode        (Retina pipeline: IoU tracking, stable ids)
      -> DinoV2Embedder                    (real frozen DINOv2 per-player vec)
      -> WorldState.from_frame             (the standardized snapshot)

Each player's symbolic position is the bbox **foot point** (bottom-centre of the
box) — where the player actually stands on the pitch, which is what we want to
predict. Appearance `entity.vec` is a genuine DINOv2-small (384-d) embedding of
the player crop, so the dynamics model can (in principle) condition motion on who
the player is.

The output JSON is a list of sequences (here, the one clip split into overlapping
windows is done downstream; this writes the single long sequence) of frames, each
frame a list of `{id, cx, cy, vec}`. Run on the Mac Studio (MPS) with the heavy
extras; it's ~minutes for a 30s clip sampled at ~6 fps.

    python examples/world_model/soccer/record.py \
        --clip ~/work/soccer-demo/raw/08fd33.mp4 \
        --weights ~/work/soccer-demo/yolo11x.pt \
        --out examples/world_model/soccer/data/soccer_states.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2

# Allow `python examples/world_model/soccer/record.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from retina import DinoV2Embedder, IoUTracker, WorldState  # noqa: E402
from retina.detect import Detection  # noqa: E402
from retina.nodes import DetectorNode, TrackerNode  # noqa: E402
from retina.pipeline import Pipeline  # noqa: E402

DINO_SIZE = "small"
DINO_DIM = 384


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--weights", default="yolo11x.pt")
    ap.add_argument("--out", default="examples/world_model/soccer/data/soccer_states.json")
    ap.add_argument("--frames-out", default="examples/world_model/soccer/data/frames",
                    help="dir to dump the sampled BGR frames (for the renderer)")
    ap.add_argument("--step", type=int, default=5, help="sample every Nth video frame")
    ap.add_argument("--max-frames", type=int, default=120, help="cap sampled frames")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)

    # A tiny stateful detector for the Retina pipeline. We run YOLO + supervision
    # ourselves (to keep the real YOLO -> sv.Detections -> Detection.from_supervision
    # interop path) and hand the resulting Retina Detections to the pipeline via
    # this holder. The pipeline then does the tracking and (after) embedding.
    class _Holder:
        dets: list[Detection] = []

        def __call__(self, _image):
            return self.dets

    holder = _Holder()

    # Retina pipeline: detector node (our holder) + IoU tracker for stable ids.
    # min_hits=1 so a player is tracked the frame it appears; a generous IoU
    # threshold and max_missed cope with the ~6fps sampling and brief occlusions.
    pipe = Pipeline(
        [
            DetectorNode(holder),
            TrackerNode(IoUTracker(min_hits=1, iou_threshold=0.2, max_missed=3)),
        ],
        source_id="soccer",
    )
    embedder = DinoV2Embedder(size=DINO_SIZE, device=args.device, bgr=True)

    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"clip {args.clip}: {W}x{H} @ {fps:.1f}fps; sampling every {args.step} "
          f"-> ~{fps / args.step:.1f}fps, max {args.max_frames} frames")

    os.makedirs(args.frames_out, exist_ok=True)
    states: list[dict] = []
    fi = -1
    import supervision as sv

    while len(states) < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if fi % args.step:
            continue
        t = round(fi / fps, 3)

        # --- real detection -> supervision -> Retina Detection (interop path) ---
        res = model(frame, imgsz=args.imgsz, conf=args.conf, classes=[0], verbose=False)[0]
        sv_det = sv.Detections.from_ultralytics(res)
        holder.dets = Detection.from_supervision(sv_det)

        rframe = pipe.process(frame, t)        # detect + track
        embedder(rframe)                       # real DINOv2 vec onto each track
        ws = WorldState.from_frame(rframe)      # standardized snapshot

        ents = []
        for e in ws.entities:
            if e.bbox is None:
                continue
            x1, y1, x2, y2 = e.bbox
            # foot point: bottom-centre of the box = where the player stands.
            fx = (x1 + x2) / 2.0
            fy = y2
            vec = e.vec.values if e.vec is not None else None
            if vec is not None:
                vec = [round(float(v), 4) for v in vec]
            ents.append({
                "id": e.id,
                "cx": round(float(fx), 2),
                "cy": round(float(fy), 2),
                "bbox": [round(float(b), 1) for b in (x1, y1, x2, y2)],
                "vec": vec,
            })
        states.append({"t": t, "frame_idx": fi, "entities": ents})

        # save the BGR frame jpg so the renderer overlays on REAL footage
        cv2.imwrite(os.path.join(args.frames_out, f"{len(states) - 1:04d}.jpg"), frame)
        if len(states) % 10 == 0:
            print(f"  sampled {len(states)} frames (t={t:.2f}s, {len(ents)} players)")

    cap.release()

    out = {
        "spec": "retina.world_model.soccer/0.1",
        "scene": "real broadcast soccer footage; players detected+tracked, "
                 "position = bbox foot-point, appearance = frozen DINOv2-small",
        "clip": os.path.basename(args.clip),
        "W": W,
        "H": H,
        "fps": fps,
        "step": args.step,
        "vec_model": f"dinov2-{DINO_SIZE}",
        "vec_dim": DINO_DIM,
        "frames_dir": args.frames_out,
        "states": states,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(out, fp, separators=(",", ":"))
    size_kb = os.path.getsize(args.out) / 1024
    n_players = len({e["id"] for st in states for e in st["entities"]})
    print(f"wrote {args.out} ({size_kb:.0f} KB): {len(states)} frames, "
          f"{n_players} distinct player tracks")


if __name__ == "__main__":
    main()
