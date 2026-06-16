"""Validate events against retina.event/0.1 (see SPEC.md).

Pure-Python — no `jsonschema` dependency, so the core stays numpy-only. The
formal JSON Schema (for other languages / tooling) ships beside this module as
`event.schema.json` and is returned by `load_schema()`.
"""

from __future__ import annotations

from typing import Any

_REQUIRED = ("type", "t", "src")

_OPTIONAL_TYPES: dict[str, Any] = {
    "id": int,
    "label": str,
    "zone": str,
    "dur": (int, float),
    "dir": str,
    "n": int,
    "conf": (int, float),
    "box": (list, tuple),
    "by": str,
    "frame": int,
    "clip": str,
    "eid": str,
    "vec": dict,
}


def validate(event) -> list[str]:
    """Return a list of problems (empty = valid). Accepts an `Event` or a dict."""
    d = event.to_dict() if hasattr(event, "to_dict") else dict(event)
    errs: list[str] = []

    for k in _REQUIRED:
        if d.get(k) is None:
            errs.append(f"missing required field: {k!r}")

    if isinstance(d.get("type"), str) is False and "type" in d:
        errs.append("'type' must be a string")
    if "t" in d and not isinstance(d["t"], (int, float, str)):
        errs.append("'t' must be a number (epoch seconds) or RFC3339 string")
    if "src" in d and not isinstance(d["src"], str):
        errs.append("'src' must be a string")

    for k, ty in _OPTIONAL_TYPES.items():
        if d.get(k) is not None and not isinstance(d[k], ty):
            name = ty.__name__ if isinstance(ty, type) else "/".join(t.__name__ for t in ty)
            errs.append(f"'{k}' must be {name}")

    box = d.get("box")
    if isinstance(box, (list, tuple)) and len(box) != 4:
        errs.append("'box' must be [x1, y1, x2, y2]")
    conf = d.get("conf")
    if isinstance(conf, (int, float)) and not (0.0 <= conf <= 1.0):
        errs.append("'conf' must be in 0..1")

    # the latent sub-object: model + dim are required (parity with event.schema.json)
    vec = d.get("vec")
    if isinstance(vec, dict):
        if not isinstance(vec.get("model"), str):
            errs.append("'vec.model' must be a string")
        if not isinstance(vec.get("dim"), int) or isinstance(vec.get("dim"), bool):
            errs.append("'vec.dim' must be an integer")

    return errs


def is_valid(event) -> bool:
    return not validate(event)


def load_schema() -> dict:
    """The formal JSON Schema (draft 2020-12) for retina.event."""
    import json
    from importlib import resources

    return json.loads(resources.files("retina").joinpath("event.schema.json").read_text())
