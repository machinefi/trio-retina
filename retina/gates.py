"""Gates: cheap "should I even look at this frame?" signals.

A gate is `callable(image, t) -> bool`. Use it as a Retina `gate=` or a
`GateNode` to skip the detector (or, later, an expensive VLM) on uninteresting
frames — the cascade pattern that keeps cost down.
"""

from __future__ import annotations

import numpy as np


class MotionGate:
    """Look only when the frame changed from the previous one (mean abs diff)."""

    def __init__(self, thresh: float = 0.5):
        self.prev: np.ndarray | None = None
        self.thresh = thresh

    def __call__(self, image: np.ndarray, t: float) -> bool:
        if self.prev is None:
            self.prev = image
            return True
        diff = float(np.abs(image.astype(np.int16) - self.prev.astype(np.int16)).mean())
        self.prev = image
        return diff > self.thresh
