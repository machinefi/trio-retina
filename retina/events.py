"""The wire contract (`Event`) and the in-pipeline data unit (`Frame`).

`Event` follows `SPEC.md` (retina.event/0.1): tiny and flat, like a JWT —
required `type/t/src`, everything else optional, empty fields omitted on
serialize so the minimal event is three keys. Custom data goes in `ext` and is
flattened into the JSON (namespace your keys).

`Frame` is the enrichment unit (DeepStream's metadata-tree idea): each stage
populates its own field (detections, then tracks, then events) as the frame flows
through the chain. The open `user` dicts are extension slots for downstream code.

Keep this module stdlib-only — it's the schema everyone agrees on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .worldstate import Vec

SPEC = "retina.event/0.1"

BBox = tuple[float, float, float, float]


class EventType:
    """The closed primitive vocabulary for 0.1 (see SPEC.md)."""

    ZONE_ENTER = "zone.enter"
    ZONE_EXIT = "zone.exit"
    ZONE_DWELL = "zone.dwell"
    LINE_CROSS = "line.cross"
    COUNT_THRESHOLD = "count.threshold"


# Registered optional fields, in serialization order (SPEC.md).
_OPTIONAL = (
    "id", "label", "zone", "dur", "dir", "n", "conf", "box", "by", "frame", "clip", "eid", "vec"
)

# All reserved keys — custom `ext` fields may never shadow these.
_RESERVED = frozenset(("type", "t", "src", *_OPTIONAL))


@dataclass(slots=True)
class Event:
    """One thing that happened. Serializes to the minimal JWT-style form."""

    type: str
    t: float
    src: str
    id: int | None = None  # track id of the subject  # noqa: A003
    label: str | None = None
    zone: str | None = None
    dur: float | None = None
    dir: str | None = None
    n: int | None = None
    conf: float | None = None
    box: BBox | None = None
    by: str | None = None
    frame: int | None = None
    clip: str | None = None
    eid: str | None = None
    # the dual-state latent: a Vec or a plain {model, dim, dtype, ref|values} dict
    vec: Vec | dict[str, Any] | None = None
    ext: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Flat dict, null/empty fields omitted, custom `ext` merged in."""
        d: dict[str, Any] = {"type": self.type, "t": self.t, "src": self.src}
        for k in _OPTIONAL:
            v = getattr(self, k)
            if v is None:
                continue
            if k == "box":
                d[k] = list(v)
            elif k == "vec":
                d[k] = v.to_dict() if hasattr(v, "to_dict") else v  # Vec or dict
            else:
                d[k] = v
        # custom fields never shadow reserved schema keys
        for k, v in self.ext.items():
            if k not in _RESERVED:
                d[k] = v
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), default=str)

    def __repr__(self) -> str:
        # Concise, information-dense: the type up front, then only the salient
        # non-None fields. Avoids the noisy auto-repr full of `None`s.
        parts: list[str] = []
        if self.id is not None:
            parts.append(f"id={self.id}")
        if self.label is not None:
            parts.append(f"label={self.label!r}")
        if self.zone is not None:
            parts.append(f"zone={self.zone!r}")
        if self.dir is not None:
            parts.append(f"dir={self.dir!r}")
        if self.dur is not None:
            parts.append(f"dur={self.dur}")
        if self.n is not None:
            parts.append(f"n={self.n}")
        parts.append(f"t={self.t}")
        tail = (" " + " ".join(parts)) if parts else ""
        return f"Event({self.type}{tail})"


@dataclass(slots=True)
class Frame:
    """Append-only enrichment unit flowing through the pipeline.

    Stages attach to it: the detector fills `detections`, the tracker fills
    `tracks`, the rules fill `events`. `user` is an open extension slot.
    """

    frame_num: int
    src: str
    t: float
    image: Any = None  # raw frame (numpy array) — what detector/gate/enricher read
    width: int = 0
    height: int = 0
    detections: list = field(default_factory=list)  # list[Detection]
    tracks: list = field(default_factory=list)  # list[Track]
    events: list = field(default_factory=list)  # list[Event]
    user: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        # Counts, not the full contents — a Frame mid-pipeline can hold dozens of
        # detections/tracks/events plus a raw image array.
        return (
            f"Frame(#{self.frame_num} src={self.src!r} t={self.t} "
            f"dets={len(self.detections)} tracks={len(self.tracks)} events={len(self.events)})"
        )
