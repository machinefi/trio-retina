"""Pipeline: wire nodes into a graph that turns frames into events.

Three ways to build the same thing — pick your altitude:

1. `|` composition (LCEL-style, recommended):
       pipe = YoloDetector("yolo11n.pt") | IoUTracker() | ZoneRule(dock) | JsonlSink("e.jsonl")

2. explicit list:
       pipe = Pipeline([DetectorNode(yolo), TrackerNode(), RuleNode(ZoneRule(dock))])

3. declarative workflow file ("n8n without a GUI"):
       pipe = Pipeline.from_json("workflow.json")

`Retina(detector=..., rules=[...])` is sugar over a Pipeline for the common case.
Every node enriches the append-only `Frame`; `run()` streams the events out.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

from .events import Event, Frame
from .nodes import DetectorNode, GateNode, Node, RuleNode, SinkNode, TrackerNode


def to_node(x) -> Node:
    """Coerce a step into a Node: pass Nodes through, auto-wrap pipeable domain
    objects (detector/tracker/rule/sink) via their `to_node()`."""
    if isinstance(x, Node):
        return x
    tn = getattr(x, "to_node", None)
    if tn is not None:
        return tn()
    raise TypeError(
        f"{x!r} is not a pipeline step; wrap a raw function in "
        "DetectorNode/GateNode/EnricherNode/SinkNode"
    )


class Pipeline:
    """A linear chain of nodes. Each frame flows through every node in order; a
    node returning None drops the frame (the rest of the chain is skipped)."""

    def __init__(self, nodes: Iterable, *, source_id: str = "cam"):
        self.nodes: list[Node] = [to_node(n) for n in nodes]
        self.source_id = source_id
        self._i = 0

    def __or__(self, other) -> "Pipeline":
        return Pipeline([*self.nodes, to_node(other)], source_id=self.source_id)

    def process(self, image, t: float) -> Frame:
        """Run one (image, timestamp) through the chain; return the enriched Frame."""
        f = Frame(frame_num=self._i, src=self.source_id, t=t, image=image)
        shape = getattr(image, "shape", None)
        if shape is not None and len(shape) >= 2:
            f.height, f.width = int(shape[0]), int(shape[1])
        self._i += 1
        cur: Frame | None = f
        for node in self.nodes:
            cur = node(cur)
            if cur is None:  # gated/dropped — stop the chain
                return f
        return cur

    def run(self, frames: Iterable[tuple]) -> Iterator[Event]:
        """Stream events from an iterable of (image, timestamp) pairs."""
        for image, t in frames:
            yield from self.process(image, t).events

    # --- declarative workflows ("n8n without a GUI") ---

    @classmethod
    def from_dict(cls, spec: dict) -> "Pipeline":
        by_id = {n["id"]: n for n in spec["nodes"]}
        flow = spec.get("flow") or [n["id"] for n in spec["nodes"]]
        nodes = []
        for nid in flow:
            n = by_id[nid]
            builder = _NODE_BUILDERS.get(n["type"])
            if builder is None:
                raise ValueError(f"unknown node type: {n['type']!r}")
            nodes.append(builder(n))
        return cls(nodes, source_id=spec.get("source_id", "cam"))

    @classmethod
    def from_json(cls, path: str) -> "Pipeline":
        import json

        with open(path) as fp:
            return cls.from_dict(json.load(fp))


class Retina:
    """Sugar over Pipeline for the common detector -> tracker -> rules case."""

    def __init__(
        self,
        source_id: str,
        detector,
        rules: Iterable,
        *,
        tracker=None,
        sinks: Iterable[Callable[[Event], None]] = (),
        gate: Callable | None = None,
    ):
        nodes: list[Node] = []
        if gate is not None:
            nodes.append(GateNode(gate))
        nodes.append(DetectorNode(detector))
        nodes.append(TrackerNode(tracker))
        nodes.extend(RuleNode(r) for r in rules)
        nodes.extend(SinkNode(s) for s in sinks)
        self.source_id = source_id
        self._pipe = Pipeline(nodes, source_id=source_id)

    def process(self, image, t: float) -> Frame:
        return self._pipe.process(image, t)

    def step(self, image, t: float) -> list[Event]:
        return self.process(image, t).events

    def run(self, frames: Iterable[tuple]) -> Iterator[Event]:
        return self._pipe.run(frames)


# --- node-type registry for declarative workflows ---


def _classes(spec) -> set[str] | None:
    c = spec.get("classes")
    return set(c) if c else None


def _yolo(spec) -> Node:
    from .detect import YoloDetector

    return DetectorNode(
        YoloDetector(
            spec.get("weights", "yolo11n.pt"),
            classes=_classes(spec),
            min_confidence=spec.get("min_confidence", 0.25),
        )
    )


def _iou_tracker(spec) -> Node:
    from .track import IoUTracker

    kw = {k: spec[k] for k in ("iou_threshold", "max_missed", "min_hits") if k in spec}
    return TrackerNode(IoUTracker(**kw))


def _zone_rule(spec) -> Node:
    from .rules import ZoneRule
    from .zones import Zone

    zone = Zone(spec["id"], [tuple(p) for p in spec["zone"]], normalized=spec.get("normalized", True))
    return RuleNode(ZoneRule(zone, classes=_classes(spec), dwell_s=spec.get("dwell_s")))


def _line_rule(spec) -> Node:
    from .rules import LineRule
    from .zones import Line

    line = Line(spec["id"], tuple(spec["a"]), tuple(spec["b"]), normalized=spec.get("normalized", True))
    return RuleNode(LineRule(line, classes=_classes(spec)))


def _count_rule(spec) -> Node:
    from .rules import CountRule
    from .zones import Zone

    zone = None
    if spec.get("zone"):
        zone = Zone(spec["id"], [tuple(p) for p in spec["zone"]], normalized=spec.get("normalized", True))
    return RuleNode(
        CountRule(
            threshold=spec["threshold"],
            classes=_classes(spec),
            zone=zone,
            comparator=spec.get("comparator", ">="),
        )
    )


def _motion_gate(spec) -> Node:
    from .gates import MotionGate

    return GateNode(MotionGate(thresh=spec.get("thresh", 0.5)))


def _jsonl(spec) -> Node:
    from .export import JsonlSink

    return SinkNode(JsonlSink(spec["path"]))


def _webhook(spec) -> Node:
    from .export import WebhookSink

    return SinkNode(WebhookSink(spec["url"]))


_NODE_BUILDERS: dict[str, Callable[[dict], Node]] = {
    "yolo": _yolo,
    "iou_tracker": _iou_tracker,
    "zone_rule": _zone_rule,
    "line_rule": _line_rule,
    "count_rule": _count_rule,
    "motion_gate": _motion_gate,
    "jsonl": _jsonl,
    "webhook": _webhook,
}


def register_node(type_name: str, builder: Callable[[dict], Node]) -> None:
    """Register a custom node type for declarative `from_json` workflows."""
    _NODE_BUILDERS[type_name] = builder
