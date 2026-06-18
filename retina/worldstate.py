"""`WorldState` — the assembled snapshot (entities now), beside the event stream.

Where `Event` is a *transition* ("a track crossed the line"), a `WorldState` is
the *state* it transitioned into: the set of entities present at one instant,
their typed relations, and an optional scene vector. The two coexist — events
are the deltas, the WorldState is the frame they apply to.

Same dual-channel rule as `Event` (see SPEC.md "The latent channel"): every
entity keeps a **symbolic core** (readable, model-agnostic) and an *optional*
model-tagged `vec` latent — never collapsed into one another. Serialization is
JWT-minimal exactly like `Event.to_dict`: null/empty fields are dropped, so the
smallest WorldState is just `{src, t}`.

Stdlib-only, like `events.py` — this is schema everyone agrees on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

SPEC = "retina.worldstate/0.1"

BBox = tuple[float, float, float, float]

_VEC_FIELDS = ("model", "dim", "dtype", "ref", "values")


def _vec_from_raw(raw: Any) -> Vec | None:
    """Build a `Vec` from a possibly-foreign dict: keep only known fields
    (ignore extras), and degrade to `None` if the required `model`/`dim` are
    missing or malformed — a foreign/future producer never kills the frame."""
    if not isinstance(raw, dict):
        return None
    if not isinstance(raw.get("model"), str) or not isinstance(raw.get("dim"), int) or isinstance(
        raw.get("dim"), bool
    ):
        return None
    return Vec(**{k: raw[k] for k in _VEC_FIELDS if k in raw})


@dataclass(slots=True)
class Vec:
    """A model-tagged latent. Small vectors ride `values` inline; large or
    re-embeddable ones ride `ref` by reference. Always tagged `{model, dim}`."""

    model: str
    dim: int
    dtype: str = "fp32"
    ref: str | None = None
    values: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"model": self.model, "dim": self.dim, "dtype": self.dtype}
        if self.ref is not None:
            d["ref"] = self.ref
        if self.values is not None:
            d["values"] = list(self.values)
        return d

    def __repr__(self) -> str:
        carrier = ""
        if self.ref is not None:
            carrier = f" ref={self.ref!r}"
        elif self.values is not None:
            carrier = " values=inline"
        return f"Vec(model={self.model!r} dim={self.dim}{carrier})"


@dataclass(slots=True)
class Entity:
    """One thing present in the scene: a symbolic core (+ optional latent `vec`)."""

    id: str  # noqa: A003
    type: str  # noqa: A003
    bbox: BBox | None = None
    conf: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    vec: Vec | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "type": self.type}
        if self.bbox is not None:
            d["bbox"] = list(self.bbox)
        if self.conf is not None:
            d["conf"] = self.conf
        if self.attrs:
            d["attrs"] = self.attrs
        if self.vec is not None:
            d["vec"] = self.vec.to_dict()
        return d

    def __repr__(self) -> str:
        parts = [f"id={self.id!r}", f"type={self.type!r}"]
        if self.conf is not None:
            parts.append(f"conf={self.conf:.2f}")
        if self.vec is not None:
            parts.append("vec")
        return f"Entity({' '.join(parts)})"


@dataclass(slots=True)
class Relation:
    """A typed, directed relation between two entities (`subj` -predicate-> `obj`).

    `family` is an optional coarse grouping (spatial / social / functional …)
    above the specific `predicate`."""

    subj: str
    obj: str  # noqa: A003
    predicate: str
    family: str | None = None
    conf: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"subj": self.subj, "obj": self.obj, "predicate": self.predicate}
        if self.family is not None:
            d["family"] = self.family
        if self.conf is not None:
            d["conf"] = self.conf
        return d

    def __repr__(self) -> str:
        return f"Relation({self.subj!r} -{self.predicate}-> {self.obj!r})"


@dataclass(slots=True)
class WorldState:
    """The assembled snapshot: entities present, their relations, scene latent."""

    src: str
    t: float
    frame: int | None = None
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    scene: Vec | None = None
    user: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Minimal dict, null/empty fields omitted — the smallest is {src, t}."""
        d: dict[str, Any] = {"src": self.src, "t": self.t}
        if self.frame is not None:
            d["frame"] = self.frame
        if self.entities:
            d["entities"] = [e.to_dict() for e in self.entities]
        if self.relations:
            d["relations"] = [r.to_dict() for r in self.relations]
        if self.scene is not None:
            d["scene"] = self.scene.to_dict()
        if self.user:
            d["user"] = self.user
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), default=str)

    def __repr__(self) -> str:
        parts = [f"src={self.src!r}", f"t={self.t}", f"entities={len(self.entities)}"]
        if self.relations:
            parts.append(f"relations={len(self.relations)}")
        if self.scene is not None:
            parts.append("scene")
        return f"WorldState({' '.join(parts)})"

    @classmethod
    def from_frame(cls, frame: Any) -> WorldState:
        """Assemble a WorldState from a `Frame`: each track becomes an entity.

        Maps the symbolic core (id/type/bbox/conf) straight off the track; if a
        per-object latent was attached upstream (in `track.user["vec"]` as a
        dict), it rides along as the entity's `vec`. A scene-level latent (e.g. a
        frozen V-JEPA scene encoder) attaches symmetrically: if
        `frame.user["scene"]` is a dict, it lifts onto `ws.scene`. Relations
        default empty — filled by a higher stage (a relation extractor)."""
        entities: list[Entity] = []
        for trk in frame.tracks:
            raw = trk.user.get("vec") if isinstance(trk.user, dict) else None
            vec = _vec_from_raw(raw)
            entities.append(
                Entity(
                    id=str(trk.track_id),
                    type=trk.label,
                    bbox=trk.bbox,
                    conf=trk.confidence,
                    vec=vec,
                )
            )
        fuser = frame.user if isinstance(getattr(frame, "user", None), dict) else {}
        raw_scene = fuser.get("scene")
        scene = _vec_from_raw(raw_scene)
        return cls(
            src=frame.src,
            t=frame.t,
            frame=frame.frame_num,
            entities=entities,
            scene=scene,
        )
