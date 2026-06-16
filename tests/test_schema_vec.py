"""Unit tests for the vec sub-object validation and load_schema parity."""

from retina import Event, load_schema, validate


def _vec_event(vec):
    return {"type": "zone.enter", "t": 1.0, "src": "cam", "vec": vec}


def test_vec_missing_model_and_dim_errors():
    errs = validate(_vec_event({"dtype": "fp32"}))  # no model, no dim
    assert any("vec.model" in e for e in errs)
    assert any("vec.dim" in e for e in errs)


def test_vec_non_int_dim_errors():
    errs = validate(_vec_event({"model": "m", "dim": 12.5}))  # dim must be int
    assert any("vec.dim" in e for e in errs)
    assert not any("vec.model" in e for e in errs)


def test_vec_bool_dim_is_not_int():
    # bool is a subclass of int but must be rejected as a dimension.
    errs = validate(_vec_event({"model": "m", "dim": True}))
    assert any("vec.dim" in e for e in errs)


def test_valid_vec_passes():
    errs = validate(_vec_event({"model": "v-jepa2", "dim": 1024}))
    assert errs == []


def test_valid_vec_event_object_passes():
    e = Event(type="zone.enter", t=1.0, src="cam", vec={"model": "m", "dim": 256})
    assert validate(e) == []


def test_load_schema_parity_smoke():
    s = load_schema()
    assert isinstance(s, dict)
    assert s["required"] == ["type", "t", "src"]
    assert "vec" in s["properties"]
    # the formal schema mirrors the python validator: vec requires model + dim
    assert s["properties"]["vec"]["required"] == ["model", "dim"]
