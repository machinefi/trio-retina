"""End-to-end Signal -> Event test with a scripted detector (no model needed)."""

import numpy as np

from retina import CountRule, Event, IoUTracker, Line, LineRule, Retina, Zone, ZoneRule
from retina.detect import Detection
from retina.events import EventType


class _ScriptedDetector:
    def __init__(self, xs):
        self._xs = xs
        self._i = 0

    def __call__(self, frame):
        if self._i >= len(self._xs):
            return []
        x = self._xs[self._i]
        self._i += 1
        if x is None:
            return []
        return [Detection(label="person", bbox=(x - 10, 40, x + 10, 60), confidence=0.9)]


def _run(rules, xs, *, min_hits=2, **kw):
    cam = Retina(
        source_id="cam",
        detector=_ScriptedDetector(xs),
        tracker=IoUTracker(min_hits=min_hits),
        rules=rules,
        **kw,
    )
    frames = [(np.zeros((100, 100, 3), dtype=np.uint8), float(i)) for i in range(len(xs))]
    return list(cam.run(frames))


def test_zone_enter_dwell_exit():
    dock = Zone("dock", [(40, 0), (60, 0), (60, 100), (40, 100)])
    xs = list(range(0, 102, 6))  # inside zone at 42,48,54
    events = _run([ZoneRule(dock, classes={"person"}, dwell_s=2.0)], xs)
    types = [e.type for e in events]
    assert EventType.ZONE_ENTER in types
    assert EventType.ZONE_DWELL in types
    assert EventType.ZONE_EXIT in types
    assert types.index(EventType.ZONE_ENTER) < types.index(EventType.ZONE_EXIT)
    dwell = next(e for e in events if e.type == EventType.ZONE_DWELL)
    assert dwell.dur >= 2.0
    assert dwell.label == "person"
    assert dwell.zone == "dock"


def test_line_cross_direction():
    tripwire = Line("door", (50, 0), (50, 100))
    xs = list(range(0, 102, 6))
    events = _run([LineRule(tripwire, classes={"person"})], xs)
    crosses = [e for e in events if e.type == EventType.LINE_CROSS]
    assert len(crosses) == 1
    assert crosses[0].dir == "a_to_b"


def test_count_threshold_edge_triggered():
    xs = list(range(0, 102, 6))
    events = _run([CountRule(threshold=1, classes={"person"})], xs)
    counts = [e for e in events if e.type == EventType.COUNT_THRESHOLD]
    assert len(counts) == 1  # one person ever present -> fires once, not per-frame
    assert counts[0].n == 1


def test_normalized_zone_autoscales():
    # Zone authored in 0..1; pipeline backfills the 100x100 frame size.
    dock = Zone("dock", [(0.4, 0), (0.6, 0), (0.6, 1), (0.4, 1)], normalized=True)
    xs = list(range(0, 102, 6))
    events = _run([ZoneRule(dock, classes={"person"})], xs)
    types = [e.type for e in events]
    assert EventType.ZONE_ENTER in types
    assert EventType.ZONE_EXIT in types


def test_class_filter_excludes_other_labels():
    dock = Zone("dock", [(40, 0), (60, 0), (60, 100), (40, 100)])
    xs = list(range(0, 102, 6))
    events = _run([ZoneRule(dock, classes={"car"})], xs)
    assert events == []


def test_gate_skips_detection():
    dock = Zone("dock", [(40, 0), (60, 0), (60, 100), (40, 100)])
    xs = list(range(0, 102, 6))
    # gate that always says "don't look" -> zero events
    events = _run([ZoneRule(dock, classes={"person"})], xs, gate=lambda f, t: False)
    assert events == []


def test_pipeline_stamps_src_on_events():
    dock = Zone("dock", [(40, 0), (60, 0), (60, 100), (40, 100)])
    xs = list(range(0, 102, 6))
    events = _run([ZoneRule(dock, classes={"person"})], xs)  # no src passed to rule
    assert events and all(e.src == "cam" for e in events)


# --- Pipeline / composition ---


def test_pipe_operator_matches_retina():
    from retina import CallableDetector, IoUTracker

    dock = Zone("dock", [(40, 0), (60, 0), (60, 100), (40, 100)])
    xs = list(range(0, 102, 6))
    pipe = (
        CallableDetector(_ScriptedDetector(xs))
        | IoUTracker(min_hits=2)
        | ZoneRule(dock, classes={"person"}, dwell_s=2.0)
    )
    frames = [(np.zeros((100, 100, 3), np.uint8), float(i)) for i in range(len(xs))]
    via_pipe = [e.type for e in pipe.run(frames)]
    via_retina = [e.type for e in _run([ZoneRule(dock, classes={"person"}, dwell_s=2.0)], xs)]
    assert via_pipe == via_retina and via_pipe


def test_from_dict_builds_pipeline(tmp_path):
    from retina import Pipeline

    spec = {
        "source_id": "cam",
        "nodes": [
            {"id": "gate", "type": "motion_gate", "thresh": 0.5},
            {"id": "dock", "type": "zone_rule", "zone": [[0.4, 0], [0.6, 0], [0.6, 1], [0.4, 1]], "dwell_s": 2.0},
            {"id": "out", "type": "jsonl", "path": str(tmp_path / "e.jsonl")},
        ],
        "flow": ["gate", "dock", "out"],
    }
    pipe = Pipeline.from_dict(spec)
    assert [type(n).__name__ for n in pipe.nodes] == ["GateNode", "RuleNode", "SinkNode"]
    assert pipe.source_id == "cam"


def test_enricher_node_populates_user():
    from retina import CallableDetector, EnricherNode, IoUTracker, Pipeline

    xs = list(range(0, 102, 6))
    pipe = Pipeline(
        [
            CallableDetector(_ScriptedDetector(xs)),
            IoUTracker(min_hits=2),
            EnricherNode(lambda f: {"note": len(f.tracks)}, ),
        ]
    )
    f = pipe.process(np.zeros((100, 100, 3), np.uint8), 0.0)
    assert "note" in f.user


# --- SPEC conformance (retina.event/0.1) ---


def test_minimal_event_is_three_fields():
    e = Event(type="line.cross", t=1.0, src="cam_01")
    assert e.to_dict() == {"type": "line.cross", "t": 1.0, "src": "cam_01"}


def test_optional_fields_omitted_when_none():
    e = Event(type="zone.dwell", t=1.0, src="cam", id=42, label="person", dur=31.0)
    d = e.to_dict()
    assert d == {"type": "zone.dwell", "t": 1.0, "src": "cam", "id": 42, "label": "person", "dur": 31.0}
    assert "conf" not in d and "box" not in d


def test_ext_is_flattened():
    e = Event(type="zone.enter", t=1.0, src="cam", ext={"acme.shift": "night"})
    assert e.to_dict()["acme.shift"] == "night"


def test_box_serializes_as_list():
    e = Event(type="zone.enter", t=1.0, src="cam", box=(1, 2, 3, 4))
    s = e.to_json()
    assert '"box":[1,2,3,4]' in s
