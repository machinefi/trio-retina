"""Nodes: the pipeline building blocks.

A Node takes a `Frame`, enriches it (each stage populates its own field), and
returns it — or returns None to drop the frame (a gate skipping downstream work).
Each kind wraps one concern, so you compose them like n8n nodes (no GUI) or LCEL:

    DetectorNode(yolo) | TrackerNode() | GateNode(motion) | ZoneRule(dock) | SinkNode(jsonl)

The shipped detector/tracker/rule/sink objects auto-wrap into the right Node, so
you usually write `yolo | IoUTracker() | ZoneRule(dock) | JsonlSink(...)` and
only reach for an explicit Node to wrap your own raw function.
"""

from __future__ import annotations

from .compose import Pipeable
from .events import Frame
from .track import IoUTracker
from .worldstate import WorldState


class Node(Pipeable):
    """A pipeline step: Frame -> Frame (or None to drop the frame)."""

    def to_node(self) -> "Node":
        return self

    def __call__(self, frame: Frame) -> Frame | None:
        raise NotImplementedError


class DetectorNode(Node):
    """Run a detector on the frame image; fill `frame.detections`."""

    def __init__(self, detector):
        self.detector = detector

    def __call__(self, frame: Frame) -> Frame:
        frame.detections = self.detector(frame.image)
        return frame


class TrackerNode(Node):
    """Give detections identity over time; fill `frame.tracks`."""

    def __init__(self, tracker=None):
        self.tracker = tracker or IoUTracker()

    def __call__(self, frame: Frame) -> Frame:
        frame.tracks = self.tracker.update(frame.detections, frame.t)
        return frame


class RuleNode(Node):
    """Run an event rule over the tracks; append to `frame.events`."""

    def __init__(self, rule):
        self.rule = rule

    def __call__(self, frame: Frame) -> Frame:
        bind = getattr(self.rule, "bind_frame_size", None)
        if bind is not None and frame.width and frame.height:
            bind(frame.width, frame.height)
        events = self.rule.update(frame.tracks, frame.t, frame.frame_num)
        for ev in events:
            if not ev.src:  # rule left it unset (None / "") -> stamp the frame source
                ev.src = frame.src
        frame.events.extend(events)
        return frame


class GateNode(Node):
    """Drop the frame (skip everything downstream) when the gate says don't look."""

    def __init__(self, gate):
        self.gate = gate

    def __call__(self, frame: Frame) -> Frame | None:
        return frame if self.gate(frame.image, frame.t) else None


class EnricherNode(Node):
    """Run a function on the frame and merge its result into `frame.user`.

    The seam for a VLM describe, a classifier, or a V-JEPA novelty score. `fn`
    takes the Frame and returns a dict (merged into `frame.user`) or any value
    (stored under `key`)."""

    def __init__(self, fn, *, key: str | None = None):
        self.fn = fn
        self.key = key

    def __call__(self, frame: Frame) -> Frame:
        out = self.fn(frame)
        if out is not None:
            if self.key is not None:
                frame.user[self.key] = out
            elif isinstance(out, dict):
                frame.user.update(out)
        return frame


class SinkNode(Node):
    """Emit each event on the frame to a sink (jsonl/webhook/kafka/...)."""

    def __init__(self, sink):
        self.sink = sink

    def __call__(self, frame: Frame) -> Frame:
        for ev in frame.events:
            self.sink(ev)
        return frame


class WorldStateNode(Node):
    """Assemble a `WorldState` snapshot from the frame's tracks and store it on
    `frame.user[key]`, so the *state* channel flows through the same composable
    pipeline as events. Read it off `frame.user` or via `Pipeline.run_states()`."""

    def __init__(self, *, key: str = "worldstate"):
        self.key = key

    def __call__(self, frame: Frame) -> Frame:
        frame.user[self.key] = WorldState.from_frame(frame)
        return frame
