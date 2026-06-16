"""Unit tests for declarative Pipeline.from_json (node-builder registry)."""

import json

import numpy as np

from retina import CallableDetector, IoUTracker, Pipeline
from retina.detect import Detection
from retina.events import EventType


class _ScriptedDetector:
    def __init__(self, xs):
        self._xs = xs
        self._i = 0

    def __call__(self, image):
        if self._i >= len(self._xs):
            return []
        x = self._xs[self._i]
        self._i += 1
        return [Detection(label="person", bbox=(x - 10, 40, x + 10, 60), confidence=0.9)]


def test_from_json_constructs_and_runs(tmp_path):
    spec = {
        "source_id": "cam",
        "nodes": [
            {"id": "trk", "type": "iou_tracker", "min_hits": 2, "iou_threshold": 0.3},
            {
                "id": "dock",
                "type": "zone_rule",
                "zone": [[0.4, 0], [0.6, 0], [0.6, 1], [0.4, 1]],
                "classes": ["person"],
            },
        ],
        "flow": ["trk", "dock"],
    }
    wf = tmp_path / "workflow.json"
    wf.write_text(json.dumps(spec))

    built = Pipeline.from_json(str(wf))
    assert [type(n).__name__ for n in built.nodes] == ["TrackerNode", "RuleNode"]
    assert built.source_id == "cam"
    # the iou_tracker node really carries an IoUTracker
    assert isinstance(built.nodes[0].tracker, IoUTracker)

    # prepend a scripted detector and drive synthetic detections through it
    xs = list(range(0, 102, 6))  # walks left->right across the dock zone
    pipe = Pipeline(
        [CallableDetector(_ScriptedDetector(xs)), *built.nodes],
        source_id=built.source_id,
    )
    frames = [(np.zeros((100, 100, 3), np.uint8), float(i)) for i in range(len(xs))]
    types = [e.type for e in pipe.run(frames)]
    assert EventType.ZONE_ENTER in types
    assert EventType.ZONE_EXIT in types


def test_from_dict_unknown_node_type_raises(tmp_path):
    spec = {"nodes": [{"id": "x", "type": "does_not_exist"}]}
    try:
        Pipeline.from_dict(spec)
    except ValueError as e:
        assert "does_not_exist" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown node type")
