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

from .compose import Pipeable
from .detect import Detection
from .geometry import BBox, Point, centroid, iou


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
        unmatched = set(range(len(detections)))
        # Greedy match: best (track, det) IoU pairs first.
        pairs: list[tuple[float, int, int]] = []
        for ti, trk in enumerate(self._tracks):
            for di, det in enumerate(detections):
                if det.label != trk.label:
                    continue
                score = iou(trk.bbox, det.bbox)
                if score >= self._iou_threshold:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)

        matched_tracks: set[int] = set()
        for score, ti, di in pairs:
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

        # Age unmatched tracks; drop the stale.
        for ti, trk in enumerate(self._tracks):
            if ti not in matched_tracks:
                trk.missed += 1
        self._tracks = [t_ for t_ in self._tracks if t_.missed <= self._max_missed]

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
    ID stability through occlusion than IoUTracker. `pip install 'retina-sdk[norfair]'`.

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
                "NorfairTracker needs norfair. Install with: pip install 'retina-sdk[norfair]'"
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
