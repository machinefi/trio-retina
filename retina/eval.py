"""Evaluate an event stream against a reference (oracle) stream.

Domain-agnostic because every pipeline speaks the same `retina.event` format:
match predicted events to reference events that share key fields (type/zone/dir)
and fall within a time tolerance, then report precision / recall / F1. This is
the objective an auto-tuner maximizes when fitting a pipeline to a few
expensive oracle labels.
"""

from __future__ import annotations

from typing import Any


def _get(e, k: str) -> Any:
    return e.get(k) if isinstance(e, dict) else getattr(e, k, None)


def match_events(
    pred: list,
    ref: list,
    *,
    time_tol: float = 2.0,
    keys: tuple[str, ...] = ("type", "zone", "dir"),
) -> tuple[int, int, int]:
    """Greedy nearest-in-time matching. Returns (tp, fp, fn).

    A predicted event matches an unused reference event with identical `keys`
    and the smallest |t_pred - t_ref| within `time_tol`."""
    ref_used = [False] * len(ref)
    tp = 0
    for p in pred:
        best, best_dt = -1, None
        for i, r in enumerate(ref):
            if ref_used[i]:
                continue
            if any(_get(p, k) != _get(r, k) for k in keys):
                continue
            dt = abs(float(_get(p, "t")) - float(_get(r, "t")))
            if dt <= time_tol and (best_dt is None or dt < best_dt):
                best, best_dt = i, dt
        if best >= 0:
            ref_used[best] = True
            tp += 1
    return tp, len(pred) - tp, len(ref) - tp


def event_f1(pred: list, ref: list, **kw) -> dict:
    """Precision / recall / F1 between predicted and reference events."""
    tp, fp, fn = match_events(pred, ref, **kw)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
