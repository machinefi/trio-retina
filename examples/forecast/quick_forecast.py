"""Forecast on Retina's WorldState — runs with NO model and NO GPU.

Proves the L4 loop end-to-end: video → Retina → a stream of `WorldState`s → a
dynamics model predicts the state a few steps ahead → score against the actual
future state Retina computed from the future frames. A constant-velocity baseline
beats a no-motion baseline — i.e. the WorldState is genuinely *dynamics-ready*.

The whole point: the forecaster eats a structured WorldState, never raw pixels —
so Retina is the necessary interface. Swap `LinearForecaster` for `TDMPC2Dynamics`
(Mac Studio) behind the same seam; swap the detector for YOLO/V-JEPA — nothing
else changes.

    python examples/forecast/quick_forecast.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dynamics import LinearForecaster, forecast_error  # noqa: E402

from retina import IoUTracker, WorldState  # noqa: E402
from retina.detect import Detection  # noqa: E402
from retina.pipeline import Pipeline  # noqa: E402
from retina.nodes import DetectorNode, TrackerNode  # noqa: E402


class Walker:
    """One person walking diagonally — has velocity in x and y, so a constant-
    velocity model should clearly beat 'assume it stays put'."""

    def __init__(self):
        self.f = 0

    def __call__(self, image):
        x, y = 60 + self.f * 8, 200 + self.f * 4
        self.f += 1
        return [Detection("person", (x - 25, y - 25, x + 25, y + 25), 0.9)] if x < 600 else []


def main():
    pipe = Pipeline([DetectorNode(Walker()), TrackerNode(IoUTracker(min_hits=2))], source_id="cam")
    # Ground-truth WorldState sequence (Retina's per-frame state).
    gt = [WorldState.from_frame(pipe.process(np.zeros((400, 640, 3), np.uint8), float(i)))
          for i in range(34)]

    horizon = 3.0  # predict 3 frames ahead
    fc = LinearForecaster()
    lin_err, naive_err = [], []
    sample = None
    for ti, state in enumerate(gt):
        fc.observe(state)
        target = ti + int(horizon)
        if target >= len(gt):
            break
        actual = gt[target]
        e_lin = forecast_error(fc.predict(horizon), actual)      # constant-velocity
        e_naive = forecast_error(gt[ti], actual)                 # assume no motion
        if e_lin["mae"] is not None:
            lin_err.append(e_lin["mae"])
        if e_naive["mae"] is not None:
            naive_err.append(e_naive["mae"])
        if sample is None and e_lin["mae"] is not None:
            p = fc.predict(horizon).entities[0]
            a = actual.entities[0]
            sample = (ti, p.bbox, a.bbox)

    def mean(xs):
        return round(sum(xs) / len(xs), 2) if xs else None

    print(f"frames: {len(gt)}   horizon: {int(horizon)} frames\n")
    if sample:
        ti, pbox, abox = sample
        pc = ((pbox[0] + pbox[2]) / 2, (pbox[1] + pbox[3]) / 2)
        ac = ((abox[0] + abox[2]) / 2, (abox[1] + abox[3]) / 2)
        print(f"sample @t={ti}: predicted centroid {tuple(round(v) for v in pc)} "
              f"vs actual {tuple(round(v) for v in ac)}")
    print("\nforecast error (mean centroid px over the run):")
    print(f"  no-motion baseline:      {mean(naive_err)}")
    print(f"  constant-velocity (L4):  {mean(lin_err)}")
    better = round(mean(naive_err) - mean(lin_err), 1) if lin_err and naive_err else 0
    print(f"\nWorldState is dynamics-ready: the velocity model cut the error by "
          f"{better} px vs no-motion.\nSwap in TDMPC2Dynamics (Mac Studio) behind the same seam to go further.")


if __name__ == "__main__":
    main()
