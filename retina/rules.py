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
from .geometry import side_of_line
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


class ZoneRule(_RuleBase):
    """`zone.enter` on entry, `zone.exit` on departure, `zone.dwell` once a track
    has stayed `dwell_s` seconds inside (fires once per visit)."""

    def __init__(
        self,
        zone: Zone,
        *,
        src: str | None = None,
        classes: set[str] | None = None,
        dwell_s: float | None = None,
        frame_size: tuple[int, int] | None = None,
    ):
        self._zone = zone
        self._src = src
        self._classes = classes
        self._dwell_s = dwell_s
        self._frame_size = frame_size
        self._inside: dict[int, list] = {}  # track_id -> [entered_at, dwell_fired]

    def update(self, tracks: list[Track], t: float, frame_idx: int) -> list[Event]:
        events: list[Event] = []
        present = {trk.track_id for trk in tracks}

        for trk in tracks:
            if not _match_class(trk.label, self._classes):
                continue
            inside = self._zone.contains(trk.centroid, self._frame_size)
            state = self._inside.get(trk.track_id)

            if inside and state is None:
                self._inside[trk.track_id] = [t, False]
                events.append(self._ev(EventType.ZONE_ENTER, trk, t, frame_idx))
            elif inside and state is not None:
                entered_at, dwell_fired = state
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
                entered_at, _ = state
                del self._inside[trk.track_id]
                events.append(
                    self._ev(EventType.ZONE_EXIT, trk, entered_at, frame_idx, dur=t - entered_at)
                )

        # A track that vanished while inside also counts as an exit.
        for tid in list(self._inside):
            if tid not in present:
                entered_at, _ = self._inside.pop(tid)
                events.append(
                    Event(
                        type=EventType.ZONE_EXIT,
                        t=entered_at,
                        src=self._src,
                        id=tid,
                        zone=self._zone.zone_id,
                        dur=round(t - entered_at, 3),
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
    `a_to_b` or `b_to_a` by which side it moved toward."""

    def __init__(
        self,
        line: Line,
        *,
        src: str | None = None,
        classes: set[str] | None = None,
        frame_size: tuple[int, int] | None = None,
    ):
        self._line = line
        self._src = src
        self._classes = classes
        self._frame_size = frame_size

    def update(self, tracks: list[Track], t: float, frame_idx: int) -> list[Event]:
        events: list[Event] = []
        a, b = self._line._scaled(self._frame_size)
        for trk in tracks:
            if not _match_class(trk.label, self._classes) or trk.prev_centroid is None:
                continue
            if self._line.crossed(trk.prev_centroid, trk.centroid, self._frame_size):
                direction = "a_to_b" if side_of_line(trk.centroid, a, b) < 0 else "b_to_a"
                events.append(
                    Event(
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
                )
        return events


class CountRule(_RuleBase):
    """`count.threshold` when the number of tracked objects (optionally inside a
    zone / of given classes) crosses `threshold`. Edge-triggered: fires once when
    the predicate flips true, re-arms when it goes false."""

    def __init__(
        self,
        *,
        threshold: int,
        src: str | None = None,
        classes: set[str] | None = None,
        zone: Zone | None = None,
        comparator: str = ">=",
        frame_size: tuple[int, int] | None = None,
        emit_initial: bool = False,
    ):
        if comparator not in (">=", ">", "<=", "<"):
            raise ValueError(f"unsupported comparator: {comparator}")
        self._src = src
        self._threshold = threshold
        self._classes = classes
        self._zone = zone
        self._comparator = comparator
        self._frame_size = frame_size
        # None = establish a baseline on the first frame without firing (only
        # real False->True transitions emit). emit_initial=True fires on frame 1
        # if the predicate is already true.
        self._prev: bool | None = False if emit_initial else None

    def _count(self, tracks: list[Track]) -> int:
        n = 0
        for trk in tracks:
            if not _match_class(trk.label, self._classes):
                continue
            if self._zone is not None and not self._zone.contains(trk.centroid, self._frame_size):
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
