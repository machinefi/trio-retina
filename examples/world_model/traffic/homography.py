"""Ground-plane homography — turn image pixels into metres on the road.

Speed from a single fixed camera is a calibration problem, not a deep-learning
one. A car's tyre-contact point lives on the road *plane*; a planar homography
`H` maps that pixel `(u, v)` to a metric world coordinate `(X, Y)` in metres.
Calibrate `H` once from four correspondences whose real-world distances you know
(lane width ≈ 3.5 m, dashed-lane segment ≈ 3 m + 9 m gap, a surveyed box), and
every subsequent pixel is metres for free.

Pure numpy — no OpenCV. The 3×3 is solved by DLT (same routine the soccer radar
uses), so this stays importable in the numpy-only core.

Convention: `H @ [u, v, 1]ᵀ ∝ [X, Y, 1]ᵀ`, i.e. **pixels → metres**.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def fit_homography(src, dst) -> np.ndarray:
    """Direct linear transform for a planar homography (cv2-free).

    `src`, `dst` are matched (x, y) point lists. Returns the 3×3 `H` with
    `H @ [x, y, 1]ᵀ ∝ [u, v, 1]ᵀ`, normalized so `H[2, 2] == 1`.
    """
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    A = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(A)
    _, _, Vt = np.linalg.svd(A)
    Hm = Vt[-1].reshape(3, 3)
    return Hm / Hm[2, 2]


def apply_h(Hm: np.ndarray, pts) -> np.ndarray:
    """Apply a homography to (N, 2) points. Returns (N, 2)."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    ones = np.ones((len(pts), 1))
    hp = np.hstack([pts, ones]) @ Hm.T
    return hp[:, :2] / hp[:, 2:3]


def foot_point(bbox) -> tuple[float, float]:
    """Bottom-centre of a bbox — the tyre/ground contact point.

    `(x1, y1, x2, y2)` → `((x1 + x2) / 2, y2)`. This is the pixel that actually
    sits on the road plane, so it is the one to project to metres (the box centre
    would float above the road and inflate distances).
    """
    x1, _y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


@dataclass
class RoadCalibration:
    """A calibrated pixel→metre map for one fixed camera view.

    Build it from four image points whose real-world (X, Y) metre coordinates on
    the road plane you know (`from_correspondences`). Then `to_metres(bbox)`
    returns the vehicle's ground position in metres, ready to drop onto
    `Entity.locus`.
    """

    H: np.ndarray  # 3×3 pixels → metres

    @classmethod
    def from_correspondences(cls, image_pts, world_pts_m) -> RoadCalibration:
        """`image_pts` (pixels) ↔ `world_pts_m` (metres on the road plane)."""
        return cls(H=fit_homography(image_pts, world_pts_m))

    def to_metres(self, bbox) -> tuple[float, float]:
        """Project a bbox's foot point to metric road coordinates (X, Y)."""
        xy = apply_h(self.H, [foot_point(bbox)])[0]
        return (float(xy[0]), float(xy[1]))

    @property
    def inverse(self) -> np.ndarray:
        """Metres → pixels (handy for drawing a metric grid / trap line back)."""
        return np.linalg.inv(self.H)
