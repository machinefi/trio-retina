"""Tests for the VLM detector seam, the Norfair adapter, and event validation."""

from types import SimpleNamespace

import numpy as np
import pytest

from retina import (
    Event,
    Retina,
    VlmDetector,
    Zone,
    ZoneRule,
    is_valid,
    load_schema,
    validate,
)
from retina import event_f1
from retina.detect import Detection


# --- event_f1: compare two event streams ---


def test_event_f1_perfect_match():
    ref = [Event(type="line.cross", t=1.0, src="c", zone="road", dir="a_to_b")]
    pred = [Event(type="line.cross", t=1.4, src="c", zone="road", dir="a_to_b")]  # within tol
    m = event_f1(pred, ref, time_tol=2.0)
    assert m["f1"] == 1.0 and m["tp"] == 1 and m["fp"] == 0 and m["fn"] == 0


def test_event_f1_counts_fp_and_fn():
    ref = [Event(type="line.cross", t=1.0, src="c", zone="road", dir="in")]
    pred = [
        Event(type="line.cross", t=1.0, src="c", zone="road", dir="out"),  # wrong dir -> fp
        Event(type="line.cross", t=50.0, src="c", zone="road", dir="in"),  # too late -> fp
    ]
    m = event_f1(pred, ref, time_tol=2.0)
    assert m["tp"] == 0 and m["fp"] == 2 and m["fn"] == 1


# --- VlmDetector: any VLM as a detector via a client(image, prompt) -> dicts ---


def test_vlm_detector_maps_client_boxes():
    def client(image, prompt):
        return [
            {"label": "person", "box": [10, 10, 50, 50], "score": 0.8},
            {"label": "car", "box": [0, 0, 5, 5], "score": 0.2},
        ]

    det = VlmDetector(client, "find people and cars", classes={"person"}, min_confidence=0.5)
    out = det(np.zeros((100, 100, 3), np.uint8))
    assert len(out) == 1
    assert out[0].label == "person" and out[0].confidence == 0.8
    assert out[0].bbox == (10.0, 10.0, 50.0, 50.0)


def test_vlm_detector_handles_empty_response():
    det = VlmDetector(lambda image, prompt: None, "x")
    assert det(np.zeros((10, 10, 3), np.uint8)) == []


# --- Detection.from_supervision: ingest a Roboflow sv.Detections (duck-typed) ---


def _sv(xyxy, confidence=None, class_id=None, data=None):
    """A tiny stub mimicking sv.Detections (we never import supervision)."""
    return SimpleNamespace(
        xyxy=np.array(xyxy, dtype=float) if len(xyxy) else np.zeros((0, 4)),
        confidence=None if confidence is None else np.array(confidence, dtype=float),
        class_id=None if class_id is None else np.array(class_id),
        data=data or {},
    )


def test_from_supervision_basic_conversion():
    sv = _sv(
        [[10, 10, 50, 50], [0, 0, 5, 5]],
        confidence=[0.8, 0.2],
        class_id=[0, 1],
    )
    out = Detection.from_supervision(sv, class_names=["person", "car"])
    assert len(out) == 2
    assert out[0].label == "person" and out[0].confidence == 0.8
    assert out[0].bbox == (10.0, 10.0, 50.0, 50.0)
    assert out[1].label == "car" and out[1].confidence == 0.2


def test_from_supervision_class_name_from_data():
    sv = _sv(
        [[1, 2, 3, 4]],
        confidence=[0.9],
        class_id=[7],
        data={"class_name": np.array(["forklift"])},
    )
    out = Detection.from_supervision(sv)  # data label wins over class_id/mapping
    assert len(out) == 1 and out[0].label == "forklift"


def test_from_supervision_class_id_dict_mapping():
    sv = _sv([[1, 2, 3, 4]], confidence=[0.5], class_id=[3])
    out = Detection.from_supervision(sv, class_names={3: "dog"})
    assert out[0].label == "dog"


def test_from_supervision_confidence_none_uses_default():
    sv = _sv([[1, 2, 3, 4]], confidence=None, class_id=[0])
    out = Detection.from_supervision(sv, class_names=["person"])
    assert out[0].confidence == 1.0  # Detection's default


def test_from_supervision_class_id_none_falls_back():
    sv = _sv([[1, 2, 3, 4]], confidence=[0.4], class_id=None)
    out = Detection.from_supervision(sv)
    assert out[0].label == ""  # no class_name, no class_id


def test_from_supervision_class_id_no_mapping_uses_str():
    sv = _sv([[1, 2, 3, 4]], confidence=[0.4], class_id=[5])
    out = Detection.from_supervision(sv)  # no names provided
    assert out[0].label == "5"


def test_from_supervision_empty():
    assert Detection.from_supervision(_sv([])) == []


def test_from_supervision_class_name_shorter_than_xyxy_falls_back():
    # class_name has 1 entry but xyxy has 2 rows: row 0 uses the name, row 1 must
    # not IndexError — it falls back to the class_id path (here -> str(class_id)).
    sv = _sv(
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        confidence=[0.9, 0.8],
        class_id=[3, 9],
        data={"class_name": np.array(["forklift"])},
    )
    out = Detection.from_supervision(sv)
    assert len(out) == 2
    assert out[0].label == "forklift"
    assert out[1].label == "9"  # row 1 fell back to class_id


# --- schema / validation ---


def test_validate_minimal_event_ok():
    assert is_valid(Event(type="line.cross", t=1.0, src="cam"))
    assert validate(Event(type="line.cross", t=1.0, src="cam")) == []


def test_validate_missing_required():
    errs = validate({"type": "x", "t": 1.0})  # no src
    assert any("src" in e for e in errs)


def test_validate_bad_conf_and_box():
    errs = validate({"type": "x", "t": 1.0, "src": "c", "conf": 1.5, "box": [1, 2, 3]})
    assert any("conf" in e for e in errs)
    assert any("box" in e for e in errs)


def test_vec_dual_state_channel():
    # the optional latent channel: a model-tagged vec rides alongside the symbols
    e = Event(type="zone.enter", t=1.0, src="cam", id=42,
              vec={"model": "v-jepa2-vitl", "dim": 1024, "dtype": "fp16", "ref": "vec://x"})
    d = e.to_dict()
    assert d["vec"]["model"] == "v-jepa2-vitl" and d["vec"]["dim"] == 1024
    assert is_valid(e)  # vec is a registered optional field


def test_validate_rejects_bool_as_number():
    # isinstance(True, int) is True, so a bool must be rejected explicitly.
    assert any("n" in e for e in validate({"type": "x", "t": 1.0, "src": "c", "n": True}))


def test_validate_rejects_non_number_box_elements():
    errs = validate({"type": "x", "t": 1.0, "src": "c", "box": ["a", "b", "c", "d"]})
    assert any("box" in e for e in errs)


def test_validate_non_dict_input_fails_closed():
    # validate(123) / validate([...]) must return a problem, not raise TypeError.
    assert validate(123) == ["event must be an object"]
    assert validate([1, 2, 3]) == ["event must be an object"]


def test_load_schema_shape():
    s = load_schema()
    assert s["required"] == ["type", "t", "src"]
    assert "conf" in s["properties"] and s["additionalProperties"] is True


# --- Norfair adapter (skipped if norfair isn't installed) ---


def test_norfair_tracker_zone_events():
    pytest.importorskip("norfair")
    from retina import NorfairTracker

    class Scripted:
        def __init__(self, xs):
            self.xs = xs
            self.i = 0

        def __call__(self, image):
            if self.i >= len(self.xs):
                return []
            x = self.xs[self.i]
            self.i += 1
            return [Detection("person", (x - 10, 40, x + 10, 60), 0.9)]

    dock = Zone("dock", [(40, 0), (60, 0), (60, 100), (40, 100)])
    xs = list(range(0, 102, 6))
    cam = Retina(
        "cam",
        Scripted(xs),
        tracker=NorfairTracker(initialization_delay=1, distance_threshold=40),
        rules=[ZoneRule(dock, classes={"person"}, dwell_s=2.0)],
    )
    frames = [(np.zeros((100, 100, 3), np.uint8), float(i)) for i in range(len(xs))]
    types = [e.type for e in cam.run(frames)]
    assert "zone.enter" in types
    assert "zone.exit" in types
