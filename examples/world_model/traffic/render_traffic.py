"""Render the speed radar — draw boxes + km/h + a trap line onto the clip.

Consumes the `states.json` written by `record_traffic.py` (per-frame boxes with
metric `locus` and `speed_kmh`, plus `speed` trap events) and paints a Trio-style
dashboard over the original footage: each vehicle boxed with its live km/h, the
mid-field speed trap drawn as a line on the road, and a running count. Out as an
mp4 you can drop in a tweet.

    python render_traffic.py --clip road.mp4 --states states.json \
        --calib calib.example.json --out radar.mp4

cv2 only (lazy). The measurement is done; this just visualizes it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from homography import RoadCalibration, apply_h  # noqa: E402

# Trio palette (BGR for cv2).
INK = (14, 16, 17)
GROUND = (248, 250, 250)
OXBLOOD = (54, 76, 200)
LIVE = (250, 246, 240)


def _pill(img, cv2, text, org, scale=0.6, fg=INK, bg=GROUND, pad=6):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + pad), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, fg, 1, cv2.LINE_AA)
    return tw + 2 * pad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", default="radar.mp4")
    ap.add_argument("--limit-kmh", type=float, default=None, help="flag speeds over this")
    args = ap.parse_args()

    import cv2

    with open(args.states) as f:
        S = json.load(f)
    with open(args.calib) as f:
        c = json.load(f)
    calib = RoadCalibration.from_correspondences(c["image_pts"], c["world_pts_m"])
    trap_x = float(S.get("trap_x", 30.0))
    by_frame = {fr["frame"]: fr for fr in S["frames"]}

    cap = cv2.VideoCapture(args.clip)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or S.get("fps", 25.0)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # Speed trap as a road line spanning the lanes: world (trap_x, Y across) -> px.
    Hinv = calib.inverse
    ys = [p[1] for p in c["world_pts_m"]]
    trap_world = [(trap_x, y) for y in np.linspace(min(ys) - 20, max(ys) + 25, 40)]
    trap_px = apply_h(Hinv, trap_world).astype(int)

    # `Event.to_dict()` flattens `ext` to top level, so read kmh straight off.
    trap_times = sorted(float(ev.get("t", 0.0)) for ev in S["events"])
    i = -1
    last = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        fr = by_frame.get(i, last)
        last = fr if i in by_frame else last
        cur_t = float(fr["t"]) if fr else i / (fps or 25.0)

        # trap line on the road
        for j in range(len(trap_px) - 1):
            cv2.line(frame, tuple(trap_px[j]), tuple(trap_px[j + 1]), OXBLOOD, 2, cv2.LINE_AA)

        if fr:
            for e in fr["entities"]:
                bbox = e.get("bbox")
                if not bbox:
                    continue
                x1, y1, x2, y2 = (int(v) for v in bbox)
                v = e.get("speed_kmh")
                if v is not None and not (0 < v < 160):
                    v = None  # display guard: drop physically implausible readings
                over = args.limit_kmh is not None and v is not None and v > args.limit_kmh
                col = OXBLOOD if over else (120, 200, 90)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                if v is not None:
                    _pill(frame, cv2, f"{v:.0f} km/h", (x1, y1 - 8), 0.55,
                          fg=(255, 255, 255) if over else INK,
                          bg=OXBLOOD if over else GROUND)

        # HUD as bottom pills — clear of the camera's own title bar up top.
        # Running count: trap measurements whose event time has been reached.
        measured = sum(1 for tt in trap_times if tt <= cur_t)
        by = H - 22
        w1 = _pill(frame, cv2, "TRIO  SPEED RADAR", (24, by), 0.6, fg=INK, bg=GROUND)
        cv2.circle(frame, (24 + w1 + 6, by - 6), 5, OXBLOOD, -1)
        cnt = f"measured: {measured}"
        (cw, _), _ = cv2.getTextSize(cnt, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        _pill(frame, cv2, cnt, (W - cw - 30, by), 0.6, fg=INK, bg=GROUND)
        vw.write(frame)
    cap.release()
    vw.release()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
