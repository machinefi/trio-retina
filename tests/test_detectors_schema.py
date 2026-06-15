"""Tests for the VLM detector seam, the Norfair adapter, and event validation."""

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


# --- event_f1: the auto-tune objective ---


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
