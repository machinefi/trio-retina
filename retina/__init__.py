"""Retina — turn camera streams into event streams.

A small, model-agnostic, hardware-neutral library for the Signal -> Event layer:
one level above object detection (Supervision gives you boxes; Retina gives you
"person entered the dock and dwelled 31s"), and one level below domain judgment.

Quickstart (3 lines, any model):

    from retina import Retina, Zone, ZoneRule, YoloDetector
    from retina.sources import video_frames

    dock = Zone("dock", [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)], normalized=True)
    cam = Retina(
        source_id="cam_01",
        detector=YoloDetector("yolo11n.pt", classes={"person"}),
        rules=[ZoneRule(dock, classes={"person"}, dwell_s=30)],
    )
    for event in cam.run(video_frames("dock.mp4")):
        print(event.to_json())

Compose models like n8n / LCEL (no GUI):

    pipe = YoloDetector("yolo11n.pt") | IoUTracker() | ZoneRule(dock) | JsonlSink("e.jsonl")
"""

from __future__ import annotations

from .compose import Pipeable
from .detect import (
    CallableDetector,
    Detection,
    Detector,
    GroundingDinoDetector,
    VlmDetector,
    YoloDetector,
)
from .eval import event_f1, match_events
from .events import SPEC, Event, EventType, Frame
from .export import JsonlSink, WebhookSink, to_jsonl
from .gates import MotionGate
from .nodes import (
    DetectorNode,
    EnricherNode,
    GateNode,
    Node,
    RuleNode,
    SinkNode,
    TrackerNode,
    WorldStateNode,
)
from .pipeline import Pipeline, Retina, register_node, to_node
from .rules import CountRule, EventRule, LineRule, ZoneRule
from .schema import is_valid, load_schema, validate
from .track import IoUTracker, NorfairTracker, Track, Tracker
from .worldstate import Entity, Relation, Vec, WorldState
from .zones import Line, Zone

__version__ = "0.0.3"

__all__ = [
    # runner
    "Retina",
    "Pipeline",
    "to_node",
    "register_node",
    # data
    "SPEC",
    "Event",
    "EventType",
    "Frame",
    "Detection",
    "Track",
    # assembled state
    "WorldState",
    "Entity",
    "Relation",
    "Vec",
    # detectors / trackers
    "Detector",
    "CallableDetector",
    "YoloDetector",
    "VlmDetector",
    "GroundingDinoDetector",
    "Tracker",
    "IoUTracker",
    "NorfairTracker",
    # zones / rules
    "Zone",
    "Line",
    "EventRule",
    "ZoneRule",
    "LineRule",
    "CountRule",
    # gates
    "MotionGate",
    # nodes
    "Node",
    "DetectorNode",
    "TrackerNode",
    "RuleNode",
    "GateNode",
    "EnricherNode",
    "SinkNode",
    "WorldStateNode",
    "Pipeable",
    # sinks
    "to_jsonl",
    "JsonlSink",
    "WebhookSink",
    # schema / validation
    "validate",
    "is_valid",
    "load_schema",
    # evaluation
    "event_f1",
    "match_events",
]
