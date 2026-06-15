"""Export Retina WorldState trajectories → JSON, to train a dynamics model on.

Runs Retina (YOLO + tracker) on a video and dumps each entity's centroid track
`{id: [[t, cx, cy, w, h], ...]}`. This is the structured state a dynamics model
learns on — produced locally; the heavy training (torch + MPS) runs on the Mac
Studio against this file, with no Retina dependency there.

    python examples/forecast/export_trajectories.py /tmp/demo.mp4 /tmp/retina_traj.json
"""

import json
import os
import sys

import cv2

from retina import IoUTracker, WorldState, YoloDetector
from retina.nodes import DetectorNode, TrackerNode
from retina.pipeline import Pipeline
from retina.sources import video_frames

CLASSES = {"car", "truck", "bus", "motorcycle", "person"}
FPS, MAX_FRAMES = 5, 320


def main(path, out):
    cap = cv2.VideoCapture(path)
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    stride = max(1, round(native / FPS))

    pipe = Pipeline(
        [DetectorNode(YoloDetector("yolo11n.pt", classes=CLASSES, min_confidence=0.3)),
         TrackerNode(IoUTracker(min_hits=3, max_missed=8))],
        source_id="cam",
    )
    print(f"extracting trajectories from {os.path.basename(path)} @ {FPS}fps…", file=sys.stderr)
    traj: dict[str, list] = {}
    for img, t in video_frames(path, stride=stride, max_frames=MAX_FRAMES):
        ws = WorldState.from_frame(pipe.process(img, t))
        for e in ws.entities:
            if e.bbox is None:
                continue
            x1, y1, x2, y2 = e.bbox
            traj.setdefault(e.id, []).append(
                [round(t, 3), round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1),
                 round(x2 - x1, 1), round(y2 - y1, 1)]
            )

    traj = {k: v for k, v in traj.items() if len(v) >= 6}  # need some history
    json.dump({"fps": FPS, "wh": [1920, 1080], "traj": traj}, open(out, "w"))
    n_pts = sum(len(v) for v in traj.values())
    print(f"wrote {out}: {len(traj)} entities, {n_pts} points")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/demo.mp4",
         sys.argv[2] if len(sys.argv) > 2 else "/tmp/retina_traj.json")
