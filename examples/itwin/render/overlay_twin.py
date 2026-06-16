"""Composite the Retina event/state stream onto a render of the iTwin iModel.

This is how the demo GIF is produced HERE (headless, no GPU): the Baytown iModel
is rendered once by the iTwin.js backend (`@itwin/core-backend` exportGraphics →
software rasterizer → plant_base.png + camera.json), and this pass draws the live
Retina layer on top with the SAME camera so everything lines up:

    iModel render (Bentley)        +        Retina layer (ours)
    plant_base.png / camera.json            retina_events.json

In a real deployment this exact overlay is `viewer/src/RetinaDecorator.ts` running
live in the iTwin Viewer; this script is the offline-preview twin of that, so the
README GIF is reproducible without a GPU box.

    python overlay_twin.py            # uses the /tmp spike artifacts by default
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(__file__)
SPIKE = "/tmp/itwin-spike/backend-probe"
BASE = os.path.join(SPIKE, "plant_base.png")
CAM = os.path.join(SPIKE, "camera.json")
EVENTS = os.path.join(HERE, "..", "retina_events.json")
OUTDIR = "/tmp/twin_frames"

# --- placement: Retina ground frame (metres) -> Baytown plant ground (metres) ---
# (the per-site calibration; tuned so the monitored road lands on the open slab.)
X0, Y0, ZG = 410.0, 113.0, 0.10
S = 0.45  # Retina metres -> plant metres
YAW = np.radians(120.0)
COS, SIN = np.cos(YAW), np.sin(YAW)

TYPE_BGR = {
    "car": (235, 206, 0), "truck": (0, 150, 255), "bus": (0, 230, 255),
    "motorcycle": (255, 0, 255), "person": (120, 255, 60),
}


def plant_pt(rx: float, ry: float) -> np.ndarray:
    return np.array([X0 + S * (rx * COS - ry * SIN), Y0 + S * (rx * SIN + ry * COS), ZG])


def make_projector(cam: dict):
    ctr = np.array(cam["ctr"])
    right = np.array(cam["right"])
    up = np.array(cam["up"])
    scale, W, H = cam["scale"], cam["W"], cam["H"]

    def project(P: np.ndarray) -> tuple[int, int]:
        rel = P - ctr
        return int(round(W / 2 + rel.dot(right) * scale)), int(round(H / 2 - rel.dot(up) * scale))

    return project, W, H


def draw_frame(base, project, frame, doc, recent):
    vis = base.copy()
    H, W = vis.shape[:2]

    # monitored zone (road_rect) on the slab
    rect = doc["meta"]["world"].get("road_rect_m", [])
    if len(rect) >= 3:
        poly = np.array([project(plant_pt(x, y)) for x, y in rect], np.int32)
        ov = vis.copy()
        cv2.fillPoly(ov, [poly], (60, 180, 75))
        cv2.addWeighted(ov, 0.16, vis, 0.84, 0, vis)
        cv2.polylines(vis, [poly], True, (60, 180, 75), 2, cv2.LINE_AA)
        cv2.putText(vis, "monitored zone", tuple(poly[0] + [4, -6]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 180, 75), 1, cv2.LINE_AA)

    for e in frame["entities"]:
        col = TYPE_BGR.get(e["type"], (255, 255, 255))
        x, y = project(plant_pt(*e["world"]))
        if e.get("forecast"):
            fx, fy = project(plant_pt(*e["forecast"]["world"]))
            cv2.arrowedLine(vis, (x, y), (fx, fy), col, 2, cv2.LINE_AA, tipLength=0.3)
        inzone = e.get("zone") == "road"
        cv2.circle(vis, (x, y), 6, col, -1, cv2.LINE_AA)
        cv2.circle(vis, (x, y), 6, (255, 255, 255) if inzone else col, 2 if inzone else 1, cv2.LINE_AA)
        cv2.putText(vis, f"#{e['id']}", (x - 8, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # HUD
    cv2.rectangle(vis, (0, 0), (W, 40), (24, 22, 20), -1)
    cv2.putText(vis, "Retina  x  iTwin.js  -  live perception on the Baytown plant twin",
                (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"t={frame['t']:.1f}s", (W - 96, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, "iModel: Baytown (Bentley sample)   |   markers + forecast + alerts = Retina from a site camera   |   forecast: "
                + doc["meta"].get("forecaster", "?"), (14, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 210, 210), 1, cv2.LINE_AA)

    # event alerts panel (top-right)
    bx = W - 340
    if recent:
        cv2.rectangle(vis, (bx, 48), (W - 10, 48 + 18 * len(recent) + 10), (24, 22, 20), -1)
        cv2.putText(vis, "retina.event", (bx + 8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1, cv2.LINE_AA)
        for i, ev in enumerate(recent):
            txt = f"{ev['type']}  #{ev.get('id','?')}" + (f"  {ev['zone']}" if ev.get("zone") else "")
            cv2.putText(vis, txt, (bx + 8, 82 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (235, 235, 235), 1, cv2.LINE_AA)
    return vis


def main(test_frame: int | None = None):
    cam = json.load(open(CAM))
    doc = json.load(open(EVENTS))
    base = cv2.imread(BASE)
    project, W, H = make_projector(cam)
    os.makedirs(OUTDIR, exist_ok=True)
    frames = doc["frames"]

    if test_frame is not None:
        recent = list(frames[test_frame]["events"])[:6]
        cv2.imwrite("/tmp/twin_test.png", draw_frame(base, project, frames[test_frame], doc, recent))
        print("wrote /tmp/twin_test.png")
        return

    recent: list = []
    for i, fr in enumerate(frames):
        for ev in fr["events"]:
            recent.insert(0, ev)
        recent = recent[:6]
        cv2.imwrite(os.path.join(OUTDIR, f"f{i:04d}.png"), draw_frame(base, project, fr, doc, recent))
    print(f"wrote {len(frames)} frames to {OUTDIR}")
    gif = os.path.join(HERE, "..", "media", "retina_itwin_demo.gif")
    os.makedirs(os.path.dirname(gif), exist_ok=True)
    pal = "/tmp/twin_pal.png"
    fps = int(doc["meta"].get("fps", 5))
    src = f"{OUTDIR}/f%04d.png"
    # list args + no shell — the JSON meta is data, never a command string
    subprocess.run(["ffmpeg", "-y", "-i", src, "-vf", "palettegen=max_colors=128",
                    pal, "-loglevel", "error"], check=True)
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", src, "-i", pal,
                    "-lavfi", "scale=900:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer",
                    gif, "-loglevel", "error"], check=True)
    sz = os.path.getsize(gif) / 1e6
    print(f"wrote {gif}  ({sz:.1f} MB)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        main(test_frame=int(sys.argv[2]) if len(sys.argv) > 2 else 40)
    else:
        main()
