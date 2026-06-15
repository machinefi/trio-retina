"""Composition: the `|` operator that wires steps into a Pipeline.

Inspired by LangChain's LCEL (`prompt | model | parser`). Any pipeable step — a
detector, tracker, rule, gate, enricher, sink, or a Node — chains with `|`:

    pipe = YoloDetector("yolo11n.pt") | IoUTracker() | ZoneRule(dock) | JsonlSink("e.jsonl")

Kept in its own tiny module so domain classes can mix it in without import cycles
(the `Pipeline` import is deferred to call time).
"""

from __future__ import annotations


class Pipeable:
    """Mixin giving `|` composition. Subclasses implement `to_node()`."""

    def to_node(self):
        raise NotImplementedError

    def __or__(self, other):
        from .pipeline import Pipeline, to_node

        return Pipeline([to_node(self), to_node(other)])
