"""Tracking: turn per-frame detections into persistent tracks.

A track gives an object identity over time — the precondition for *temporal*
events (this is the same object that entered, dwelled, and left). The built-in
`IoUTracker` is dependency-free and good enough for most fixed-camera analytics.
For crowded scenes, swap in BoT-SORT/ByteTrack behind the same `Tracker`
protocol — Retina only needs `.update(detections, t) -> list[Track]`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .compose import Pipeable
from .detect import Detection
from .geometry import BBox, Point, centroid


@dataclass(slots=True)
class Track:
    """A detected object followed across frames.

    `bbox` is the tracker's current box; `det_bbox` preserves the raw detector
    box (they differ once a Kalman/DCF tracker predicts). `user` is an open
    extension slot for downstream code."""

    track_id: int
    label: str
    bbox: BBox
    confidence: float
    first_seen: float
    last_seen: float
    seen: int = 1
    missed: int = 0
    confirmed: bool = False
    prev_centroid: Point | None = None
    det_bbox: BBox | None = None
    user: dict[str, Any] = field(default_factory=dict)

    @property
    def centroid(self) -> Point:
        return centroid(self.bbox)

    @property
    def dwell_s(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)


class Tracker(Protocol):
    def update(self, detections: list[Detection], t: float) -> list[Track]: ...


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in `a` (T,4) against every box in `b` (D,4) → (T,4) × (D)
    matrix, computed with numpy broadcasting (the O(T·D) work, in C)."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0.0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0.0, None)
    inter = iw * ih
    area_a = np.clip(ax2 - ax1, 0.0, None) * np.clip(ay2 - ay1, 0.0, None)
    area_b = np.clip(bx2 - bx1, 0.0, None) * np.clip(by2 - by1, 0.0, None)
    union = area_a + area_b - inter
    return np.where(union > 0.0, inter / np.where(union > 0.0, union, 1.0), 0.0)


class IoUTracker(Pipeable):
    """Greedy IoU association — small, deterministic, zero extra deps.

    A detection matches the highest-IoU live track of the same class above
    `iou_threshold`. Tracks survive `max_missed` frames of occlusion and become
    `confirmed` after `min_hits` hits (so transient noise never fires events).
    """

    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        max_missed: int = 15,
        min_hits: int = 3,
    ):
        self._iou_threshold = iou_threshold
        self._max_missed = max_missed
        self._min_hits = min_hits
        self._tracks: list[Track] = []
        self._next_id = 1

    def to_node(self):
        from .nodes import TrackerNode

        return TrackerNode(self)

    def update(self, detections: list[Detection], t: float) -> list[Track]:
        # Vectorized greedy IoU association: build the per-label IoU matrix in
        # numpy (the O(T·D) work), then assign highest-IoU pairs first. Identical
        # semantics to a scalar greedy match, but the heavy part runs in C.
        unmatched = set(range(len(detections)))
        dets_by_label: dict[str, list[int]] = {}
        for di, det in enumerate(detections):
            dets_by_label.setdefault(det.label, []).append(di)
        tracks_by_label: dict[str, list[int]] = {}
        for ti, trk in enumerate(self._tracks):
            tracks_by_label.setdefault(trk.label, []).append(ti)

        pairs: list[tuple[float, int, int]] = []
        for label, tis in tracks_by_label.items():
            dis = dets_by_label.get(label)
            if not dis:
                continue
            tb = np.array([self._tracks[ti].bbox for ti in tis], dtype=np.float64)
            db = np.array([detections[di].bbox for di in dis], dtype=np.float64)
            m = _iou_matrix(tb, db)
            rows, cols = np.where(m >= self._iou_threshold)
            for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
                pairs.append((float(m[r, c]), tis[r], dis[c]))
        pairs.sort(reverse=True)

        matched_tracks: set[int] = set()
        for _score, ti, di in pairs:
            if ti in matched_tracks or di not in unmatched:
                continue
            trk, det = self._tracks[ti], detections[di]
            trk.prev_centroid = trk.centroid
            trk.bbox = det.bbox
            trk.det_bbox = det.bbox
            trk.confidence = det.confidence
            trk.last_seen = t
            trk.seen += 1
            trk.missed = 0
            if trk.seen >= self._min_hits:
                trk.confirmed = True
            matched_tracks.add(ti)
            unmatched.discard(di)

        # Age unmatched tracks and drop the stale in one pass.
        survivors: list[Track] = []
        for ti, trk in enumerate(self._tracks):
            if ti not in matched_tracks:
                trk.missed += 1
            if trk.missed <= self._max_missed:
                survivors.append(trk)
        self._tracks = survivors

        # Spawn new tracks for leftover detections.
        for di in unmatched:
            det = detections[di]
            self._tracks.append(
                Track(
                    track_id=self._next_id,
                    label=det.label,
                    bbox=det.bbox,
                    det_bbox=det.bbox,
                    confidence=det.confidence,
                    first_seen=t,
                    last_seen=t,
                    confirmed=self._min_hits <= 1,
                )
            )
            self._next_id += 1

        # Only surface tracks seen *this* frame. Tracks missed this frame stay in
        # the internal list (so they keep their id and can re-associate within
        # max_missed), but they must not count as "present" for occupancy/dwell.
        return [t_ for t_ in self._tracks if t_.confirmed and t_.missed == 0]


class NorfairTracker(Pipeable):
    """Norfair adapter — pure-Python Kalman tracking with re-association, better
    ID stability through occlusion than IoUTracker. `pip install 'trio-retina[norfair]'`.

    Surfaces only tracks detected *this* frame (coasting/occluded ones are kept
    internally for re-association but not returned, so occupancy/dwell stay honest)."""

    def __init__(
        self,
        *,
        distance_threshold: float = 40.0,
        hit_counter_max: int = 15,
        initialization_delay: int = 3,
        **kwargs,
    ):
        try:
            from norfair import Tracker as _NfTracker
        except ImportError as e:  # pragma: no cover - exercised only with extra
            raise ImportError(
                "NorfairTracker needs norfair. Install with: pip install 'trio-retina[norfair]'"
            ) from e
        self._nf = _NfTracker(
            distance_function="euclidean",
            distance_threshold=distance_threshold,
            hit_counter_max=hit_counter_max,
            initialization_delay=initialization_delay,
            **kwargs,
        )
        self._first_seen: dict[int, float] = {}
        self._prev_centroid: dict[int, Point] = {}

    def to_node(self):
        from .nodes import TrackerNode

        return TrackerNode(self)

    def update(self, detections: list[Detection], t: float) -> list[Track]:
        import numpy as np
        from norfair import Detection as _NfDet

        nf_dets = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            nf_dets.append(
                _NfDet(
                    points=np.array([[cx, cy]]),
                    scores=np.array([d.confidence]),
                    label=d.label,
                    data={"bbox": d.bbox, "conf": d.confidence, "label": d.label},
                )
            )

        out: list[Track] = []
        for obj in self._nf.update(nf_dets):
            live = getattr(obj, "live_points", None)
            if live is not None and not bool(np.asarray(live).any()):
                continue  # coasting (not detected this frame)
            data = obj.last_detection.data if obj.last_detection is not None else {}
            bbox = data.get("bbox")
            if bbox is None:
                continue
            tid = obj.id
            self._first_seen.setdefault(tid, t)
            trk = Track(
                track_id=tid,
                label=data.get("label") or "object",
                bbox=bbox,
                det_bbox=bbox,
                confidence=float(data.get("conf", 1.0)),
                first_seen=self._first_seen[tid],
                last_seen=t,
                confirmed=True,
            )
            trk.prev_centroid = self._prev_centroid.get(tid)
            self._prev_centroid[tid] = trk.centroid
            out.append(trk)
        return out
