"""Synthetic traffic — a numpy-only road you can run with no camera, no model.

This is the 5-minute on-ramp. It fabricates a few vehicles driving down a
straight road at *known* speeds, projects them into a fake camera view through a
perspective homography, then runs them back through the real Retina state layer
(`WorldState` + `Entity.locus`) and the speed estimator — and checks the
recovered km/h against ground truth. No footage, no YOLO, no GPU: just
`python synthetic_traffic.py`.

The point: **speed measurement is a calibration problem on top of tracked
state.** Once the boxes are metres (homography) and the state is standardized
(Retina), the "radar" is ~30 lines. Graduate to `record_traffic.py` for the same
pipeline on a real clip.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

# Allow running as a plain script from anywhere in the repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from retina import Entity, WorldState  # noqa: E402

from homography import RoadCalibration, apply_h, fit_homography  # noqa: E402

# Camera + road geometry (metres on the ground, pixels in the image).
FRAME_W, FRAME_H = 1280, 720
# Four road-plane points (X along travel, Y lateral) and where they land in the
# image — a plausible eye-level view: X=0 is near (bottom), X=60 far (top).
WORLD_QUAD = [(0.0, -6.0), (0.0, 6.0), (60.0, 6.0), (60.0, -6.0)]  # metres
IMAGE_QUAD = [(300, 690), (980, 690), (720, 150), (560, 150)]      # pixels


@dataclass
class Vehicle:
    vid: str
    label: str
    lane_y: float   # lateral position (m)
    x0: float       # start distance (m)
    kmh: float      # true speed


VEHICLES = [
    Vehicle("1", "car", lane_y=-2.2, x0=4.0, kmh=50.0),
    Vehicle("2", "car", lane_y=2.2, x0=0.0, kmh=72.0),
    Vehicle("3", "truck", lane_y=-2.2, x0=22.0, kmh=33.0),
    Vehicle("4", "car", lane_y=2.2, x0=14.0, kmh=61.0),
]


def calibration() -> RoadCalibration:
    """The pixel→metre map for this synthetic camera (what a real one calibrates once)."""
    return RoadCalibration.from_correspondences(IMAGE_QUAD, WORLD_QUAD)


def _world_to_pixel():
    """Ground metres → image pixels, for placing synthetic cars in the frame."""
    return fit_homography(WORLD_QUAD, IMAGE_QUAD)


def make_states(*, fps: float = 12.0, seconds: float = 4.0, noise_px: float = 1.5, seed: int = 0):
    """Fabricate the scene and run it through the REAL Retina state layer.

    Returns `(states, ground_truth, calib)` where:
      * `states` — `[(t, WorldState), ...]`; each entity has a pixel `bbox` and a
        metric `locus` recovered from that bbox through the calibration (exactly
        the path a real pipeline takes).
      * `ground_truth` — `{vehicle_id: true_kmh}`.
      * `calib` — the `RoadCalibration` used (hand to a renderer / trap line).
    """
    rng = np.random.default_rng(seed)
    G = _world_to_pixel()
    calib = calibration()
    n = int(fps * seconds)
    states = []
    for i in range(n):
        t = i / fps
        ents: list[Entity] = []
        for v in VEHICLES:
            x = v.x0 + (v.kmh / 3.6) * t   # metres travelled
            if not (0.0 <= x <= 58.0):
                continue
            # project the car's ground foot point to pixels, add detector jitter
            foot = apply_h(G, [(x, v.lane_y)])[0] + rng.normal(0, noise_px, 2)
            # a perspective-plausible box: width from a 0.9 m half-span on the ground
            side = apply_h(G, [(x, v.lane_y + 0.9)])[0]
            half_w = max(6.0, abs(side[0] - foot[0]))
            h = half_w * 2.6
            bbox = (foot[0] - half_w, foot[1] - h, foot[0] + half_w, foot[1])
            # recover metric ground position from the pixel box — the real path
            X, Y = calib.to_metres(bbox)
            ents.append(Entity(id=v.vid, type=v.label, bbox=bbox, conf=0.9, locus=(X, Y)))
        states.append((t, WorldState(src="synthetic_cam", t=t, entities=ents)))
    ground_truth = {v.vid: v.kmh for v in VEHICLES}
    return states, ground_truth, calib


def _main() -> None:
    from speed import estimate_speeds

    states, gt, _calib = make_states()
    trap_x = 30.0
    events = estimate_speeds(states, src="synthetic_cam", trap_x=trap_x, window=6)

    # The measurement is the speed as the car crosses the mid-field trap — the
    # well-conditioned reading a roadside radar reports. (Far-field, near the top
    # of the frame, perspective compresses pixels and accuracy degrades; that is
    # a property of single-camera speed, not a bug — so we measure at the trap.)
    trap_kmh = {ev.id: ev.ext["kmh"] for ev in events}

    print(f"Synthetic road · {len(states)} frames · speed trap at X={trap_x:.0f} m\n")
    print(f"{'id':>3} {'type':<6} {'true':>6} {'measured':>9}  error")
    for v in VEHICLES:
        m = trap_kmh.get(int(v.vid))
        if m is None:
            print(f"{v.vid:>3} {v.label:<6} {v.kmh:6.1f}   (no fix)")
        else:
            print(f"{v.vid:>3} {v.label:<6} {v.kmh:6.1f} {m:8.1f}  {m - v.kmh:+.1f} km/h")
    print("\nSpeed-trap events (retina.event):")
    for ev in events:
        print("  " + ev.to_json())


if __name__ == "__main__":
    _main()
