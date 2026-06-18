"""Tests for the bundled / generated sample assets.

`sample_events()` is fully offline (a file shipped in the wheel), so it is tested
unconditionally. `sample_video()` needs OpenCV to *write* its synthetic clip, so
those tests skip cleanly when `[video]` isn't installed — but the no-network
behavior (cache-path logic, friendly error) is always exercised.
"""

from __future__ import annotations

import json
import os

import pytest

import retina
from retina.schema import validate


def test_sample_events_path_exists_and_offline():
    p = retina.sample_events()
    assert isinstance(p, str)
    assert os.path.isfile(p)
    assert p.endswith(".jsonl")


def test_sample_events_all_valid():
    p = retina.sample_events()
    lines = [ln for ln in open(p).read().splitlines() if ln.strip()]
    assert lines, "bundled sample should be non-empty"
    for ln in lines:
        obj = json.loads(ln)  # must be valid JSON
        assert validate(obj) == [], f"bundled event failed validation: {obj}"


def test_sample_events_has_expected_event_types():
    p = retina.sample_events()
    types = {json.loads(ln)["type"] for ln in open(p) if ln.strip()}
    # the synthetic dock scene exercises the closed primitive vocabulary
    assert {"zone.enter", "zone.exit", "count.threshold"} <= types


def test_sample_video_cache_path_uses_xdg(tmp_path, monkeypatch):
    """Cache-path logic is testable without writing a video: point XDG at a temp
    dir and check the helper builds the right path under `trio-retina/`."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from retina.assets import _cache_dir

    d = _cache_dir()
    assert d == tmp_path / "trio-retina"
    assert d.is_dir()  # created on demand


def test_sample_video_missing_opencv_message(tmp_path, monkeypatch):
    """With no cached clip and OpenCV unavailable, the error names the `[video]`
    extra instead of leaking a raw ImportError."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    import builtins

    real_import = builtins.__import__

    def _no_cv2(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("No module named 'cv2'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_cv2)
    with pytest.raises(RuntimeError, match=r"\[video\]"):
        retina.sample_video()


def test_sample_video_generates_and_caches(tmp_path, monkeypatch):
    """Full synthetic-generation path — skipped if OpenCV isn't installed."""
    pytest.importorskip("cv2")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    p = retina.sample_video()
    assert os.path.isfile(p)
    assert p.endswith(".mp4")
    assert os.path.getsize(p) > 0

    # second call returns the cached file without rewriting it
    mtime = os.path.getmtime(p)
    p2 = retina.sample_video()
    assert p2 == p
    assert os.path.getmtime(p2) == mtime

    # the clip reads back through the same source plumbing the docs use
    from retina.sources import video_frames

    frames = list(video_frames(p, max_frames=5))
    assert len(frames) == 5
    assert frames[0][0].shape == (360, 640, 3)

    # no temp/staging files left behind in the cache dir
    leftovers = [f for f in os.listdir(os.path.dirname(p)) if f.startswith(".")]
    assert leftovers == []
