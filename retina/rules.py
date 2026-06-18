"""Event rules: the Signal->Event logic (model-free, deterministic).

Each rule is a small stateful machine fed the current tracks every frame; it
emits `Event`s on transitions. This is the heart of "turn cameras into event
streams". Semantic/LLM rules, anomaly judgment, and domain policy live one layer
up (the application / commercial layer) — never here. Rules emit only generic
primitives from the closed vocabulary in SPEC.md.
"""

from __future__ import annotations

from typing import Protocol

from .compose import Pipeable
from .events import Event, EventType
from .geometry import point_in_polygon, segments_intersect, side_of_line
from .track import Track
from .zones import Line, Zone


class EventRule(Protocol):
    def update(self, tracks: list[Track], t: float, frame_idx: int) -> list[Event]: ...


class _Scalable:
    """Mixin: lets the pipeline keep the current frame size on the rule, so users
    can author rules in normalized 0..1 coords and never think about resolution.
    Re-set every frame, so a stream that changes resolution stays correct."""

    _frame_size: tuple[int, int] | None

    def bind_frame_size(self, width: int, height: int) -> None:
        self._frame_size = (width, height)


class _RuleBase(_Scalable, Pipeable):
    """Base for rules: scalable (normalized coords) + pipeable (`|`)."""

    def to_node(self):
        from .nodes import RuleNode

        return RuleNode(self)


def _match_class(label: str, classes: set[str] | None) -> bool:
    return classes is None or label in classes


_ANCHORS = ("center", "feet", "head")


def _anchor_point(trk: Track, anchor: str):
    """The body-point used to test zone membership. `center` is the bbox
    centroid (default), `feet` the bottom-center, `head` the top-center."""
    if anchor == "center":
        return trk.centroid
    x1, y1, x2, y2 = trk.bbox
    cx = (x1 + x2) / 2.0
    if anchor == "feet":
        return (cx, y2)
    return (cx, y1)  # head


class ZoneRule(_RuleBase):
    """`zone.enter` on entry, `zone.exit` on departure, `zone.dwell` once a track
    has stayed `dwell_s` seconds inside (fires once per visit).

    `exit_grace_s` keeps a track logically inside until it has been out-of-zone
    or absent for that long (rides out detection blips / id flicker without a
    spurious exit; the exit `dur` is measured to the last frame seen inside).
    `anchor` picks the body-point tested against the polygon: `center` (default,
    the centroid), `feet` (bottom-center of bbox), or `head` (top-center)."""

    def __init__(
        self,
        zone: Zone,
        *,
        src: str | None = None,
        classes: set[str] | None = None,
        dwell_s: float | None = None,
        exit_grace_s: float = 0.0,
        anchor: str = "center",
        frame_size: tuple[int, int] | None = None,
    ):
        if anchor not in _ANCHORS:
            raise ValueError(f"unsupported anchor: {anchor}")
        self._zone = zone
        self._src: str = src or ""  # empty -> stamped with the frame source by RuleNode
        self._classes = classes
        self._dwell_s = dwell_s
        self._exit_grace_s = exit_grace_s
        self._anchor = anchor
        self._frame_size = frame_size
        # track_id -> [entered_at, dwell_fired, last_inside_t]. A track stays
        # logically inside until it has been out/absent for >= exit_grace_s; the
        # exit `dur` is then measured to last_inside_t, not the current t.
        self._inside: dict[int, list] = {}

    def update(self, tracks: list[Track], t: float, frame_idx: int) -> list[Event]:
        events: list[Event] = []
        present = {trk.track_id for trk in tracks}
        poly = self._zone.scaled(self._frame_size)  # scale once, reuse per track

        for trk in tracks:
            if not _match_class(trk.label, self._classes):
                continue
            inside = point_in_polygon(_anchor_point(trk, self._anchor), poly)
            state = self._inside.get(trk.track_id)

            if inside and state is None:
                self._inside[trk.track_id] = [t, False, t]
                events.append(self._ev(EventType.ZONE_ENTER, trk, t, frame_idx))
            elif inside and state is not None:
                # Back inside (possibly within grace): resume normal accounting.
                state[2] = t
                entered_at, dwell_fired, _ = state
                if (
                    self._dwell_s is not None
                    and not dwell_fired
                    and (t - entered_at) >= self._dwell_s
                ):
                    state[1] = True
                    events.append(
                        self._ev(EventType.ZONE_DWELL, trk, entered_at, frame_idx, dur=t - entered_at)
                    )
            elif not inside and state is not None:
                entered_at, _, last_inside_t = state
                if (t - last_inside_t) >= self._exit_grace_s:
                    del self._inside[trk.track_id]
                    # With no grace, the exit dates to the current frame (legacy
                    # behavior); with grace, to the last frame actually inside.
                    left_t = t if self._exit_grace_s == 0.0 else last_inside_t
                    events.append(
                        self._ev(
                            EventType.ZONE_EXIT, trk, entered_at, frame_idx,
                            dur=left_t - entered_at,
                        )
                    )
                # else: still within grace -> stays logically inside, no event.

        # A track that vanished while inside also counts as an exit, subject to
        # the same grace window (a brief id flicker does not fire a spurious exit).
        for tid in list(self._inside):
            if tid in present:
                continue
            entered_at, _, last_inside_t = self._inside[tid]
            if (t - last_inside_t) < self._exit_grace_s:
                continue  # within grace -> still logically inside
            del self._inside[tid]
            left_t = t if self._exit_grace_s == 0.0 else last_inside_t
            events.append(
                Event(
                    type=EventType.ZONE_EXIT,
                    t=entered_at,
                    src=self._src,
                    id=tid,
                    zone=self._zone.zone_id,
                    dur=round(left_t - entered_at, 3),
                    ext={"reason": "track_lost"},
                )
            )
        return events

    def _ev(self, etype, trk, t, frame_idx, dur=None) -> Event:
        return Event(
            type=etype,
            t=t,
            src=self._src,
            id=trk.track_id,
            label=trk.label,
            zone=self._zone.zone_id,
            dur=round(dur, 3) if dur is not None else None,
            conf=round(trk.confidence, 3),
            box=trk.bbox,
            frame=frame_idx,
        )


class LineRule(_RuleBase):
    """`line.cross` when a track's centroid crosses the tripwire. `dir` is
    `a_to_b` or `b_to_a` by which side it moved toward.

    Requires *tracked* input (each track carries an id and `prev_centroid`), per
    the standard — `line.cross` is meaningless without object identity.

    `min_frames` (default 1) is a jitter debounce, like Supervision's
    `LineZone.minimum_crossing_threshold`. With `min_frames=1` the rule is
    stateless and emits the instant the prev→curr centroid segment intersects
    the line (the original behavior). With `min_frames > 1`, a crossing is
    *pending* once the segment intersects, and is **confirmed and emitted only
    after** the track has stayed continuously on the new side for `min_frames`
    frames (including the crossing frame). If the track returns to the original
    side before then, the crossing is discarded as jitter and nothing is
    emitted. The event fires on the frame the crossing is confirmed, carrying
    the direction of the original crossing (and that frame's `t` / `box`)."""

    def __init__(
        self,
        line: Line,
        *,
        src: str | None = None,
        classes: set[str] | None = None,
        min_frames: int = 1,
        frame_size: tuple[int, int] | None = None,
    ):
        if min_frames < 1:
            raise ValueError(f"min_frames must be >= 1, got {min_frames}")
        self._line = line
        self._src: str = src or ""  # empty -> stamped with the frame source by RuleNode
        self._classes = classes
        self._min_frames = min_frames
        self._frame_size = frame_size
        # track_id -> [direction, side, frames_held]: an unconfirmed crossing,
        # the side (sign) the track must stay on, and how many frames it has.
        self._pending: dict[int, list] = {}

    def update(self, tracks: list[Track], t: float, frame_idx: int) -> list[Event]:
        events: list[Event] = []
        a, b = self._line.scaled(self._frame_size)  # scale once, reuse per track
        for trk in tracks:
            if not _match_class(trk.label, self._classes) or trk.prev_centroid is None:
                continue
            crossed = segments_intersect(trk.prev_centroid, trk.centroid, a, b)
            direction = "a_to_b" if side_of_line(trk.centroid, a, b) < 0 else "b_to_a"

            if self._min_frames == 1:
                if crossed:
                    events.append(self._ev(trk, t, frame_idx, direction))
                continue

            side = side_of_line(trk.centroid, a, b)
            pending = self._pending.get(trk.track_id)
            if crossed:
                # A fresh crossing (re-)arms the pending confirmation on the new
                # side; this frame counts as the first frame held.
                self._pending[trk.track_id] = [direction, side, 1]
            elif pending is not None:
                pend_dir, pend_side, held = pending
                if (side > 0) == (pend_side > 0):
                    held += 1
                    pending[2] = held
                    if held >= self._min_frames:
                        del self._pending[trk.track_id]
                        events.append(self._ev(trk, t, frame_idx, pend_dir))
                else:
                    # Bounced back to the original side before confirmation.
                    del self._pending[trk.track_id]
        return events

    def _ev(self, trk: Track, t: float, frame_idx: int, direction: str) -> Event:
        return Event(
            type=EventType.LINE_CROSS,
            t=t,
            src=self._src,
            id=trk.track_id,
            label=trk.label,
            zone=self._line.line_id,
            dir=direction,
            conf=round(trk.confidence, 3),
            box=trk.bbox,
            frame=frame_idx,
        )


class CountRule(_RuleBase):
    """`count.threshold` when the number of tracked objects (optionally inside a
    zone / of given classes) crosses `threshold`. Edge-triggered: fires once when
    the predicate flips true, re-arms when it goes false."""

    def __init__(
        self,
        threshold: int,
        *,
        src: str | None = None,
        classes: set[str] | None = None,
        zone: Zone | None = None,
        comparator: str = ">=",
        anchor: str = "center",
        frame_size: tuple[int, int] | None = None,
        emit_initial: bool = False,
    ):
        if comparator not in (">=", ">", "<=", "<"):
            raise ValueError(f"unsupported comparator: {comparator}")
        if anchor not in _ANCHORS:
            raise ValueError(f"unsupported anchor: {anchor}")
        self._src: str = src or ""  # empty -> stamped with the frame source by RuleNode
        self._threshold = threshold
        self._classes = classes
        self._zone = zone
        self._comparator = comparator
        self._anchor = anchor
        self._frame_size = frame_size
        # None = establish a baseline on the first frame without firing (only
        # real False->True transitions emit). emit_initial=True fires on frame 1
        # if the predicate is already true.
        self._prev: bool | None = False if emit_initial else None

    def _count(self, tracks: list[Track]) -> int:
        poly = self._zone.scaled(self._frame_size) if self._zone is not None else None
        n = 0
        for trk in tracks:
            if not _match_class(trk.label, self._classes):
                continue
            if poly is not None and not point_in_polygon(_anchor_point(trk, self._anchor), poly):
                continue
            n += 1
        return n

    def _hit(self, n: int) -> bool:
        c = self._comparator
        return (
            (c == ">=" and n >= self._threshold)
            or (c == ">" and n > self._threshold)
            or (c == "<=" and n <= self._threshold)
            or (c == "<" and n < self._threshold)
        )

    def update(self, tracks: list[Track], t: float, frame_idx: int) -> list[Event]:
        n = self._count(tracks)
        hit = self._hit(n)
        fire = self._prev is not None and hit and not self._prev
        self._prev = hit
        if not fire:
            return []
        return [
            Event(
                type=EventType.COUNT_THRESHOLD,
                t=t,
                src=self._src,
                n=n,
                zone=self._zone.zone_id if self._zone else None,
                frame=frame_idx,
                ext={"threshold": self._threshold, "cmp": self._comparator},
            )
        ]
