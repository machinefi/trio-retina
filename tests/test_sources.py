"""Tests for `video_frames` live-source hardening — reconnect + drop-to-latest.

All tests inject a FAKE capture object (no cv2, no real RTSP stream), so they run
on a numpy-only install. The fakes emulate the small VideoCapture surface
`video_frames` uses: `isOpened()`, `read() -> (ok, frame)`, `get(prop)`,
`release()`.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from retina.sources import _is_live_source, video_frames


def _frame(val: int) -> np.ndarray:
    return np.full((4, 4, 3), val % 256, dtype=np.uint8)


class FakeCapture:
    """Reads a scripted list of `(ok, frame)` results, then keeps returning
    `(False, None)`. `opened=False` simulates an open that never succeeds."""

    def __init__(self, results, *, opened=True, fps=30.0):
        self._results = list(results)
        self._i = 0
        self._opened = opened
        self._fps = fps
        self.released = False

    def isOpened(self):  # noqa: N802 - mirrors cv2 API
        return self._opened

    def read(self):
        if self._i < len(self._results):
            r = self._results[self._i]
            self._i += 1
            return r
        return (False, None)

    def get(self, _prop):
        return self._fps

    def release(self):
        self.released = True


# --- source classification ---------------------------------------------------


def test_rtsp_url_is_live():
    assert _is_live_source("rtsp://cam/stream", live=False)
    assert _is_live_source("rtmp://cam/stream", live=False)
    assert _is_live_source(0, live=False)  # webcam index
    assert _is_live_source("video.mp4", live=True)  # explicit flag


def test_plain_file_is_not_live():
    assert not _is_live_source("video.mp4", live=False)
    assert not _is_live_source("/path/to/clip.avi", live=False)


# --- finite-file behavior is unchanged ---------------------------------------


def test_finite_file_ends_at_eof_no_reconnect():
    """A real EOF on a file ends the generator — no reconnect, media-time ts."""
    cap = FakeCapture([(True, _frame(i)) for i in range(3)], fps=10.0)

    out = list(video_frames("clip.mp4", capture_factory=lambda s: cap))

    assert len(out) == 3
    # media-time timestamps: idx / fps
    assert [ts for _, ts in out] == [0.0, 0.1, 0.2]
    assert cap.released


def test_finite_file_stride_and_max_frames():
    cap = FakeCapture([(True, _frame(i)) for i in range(10)], fps=10.0)
    out = list(video_frames("clip.mp4", stride=2, max_frames=3, capture_factory=lambda s: cap))
    assert len(out) == 3
    # stride=2 -> idx 0,2,4 -> media-time 0.0, 0.2, 0.4
    assert [round(ts, 3) for _, ts in out] == [0.0, 0.2, 0.4]


def test_file_does_not_reconnect_even_if_reconnect_true():
    """`reconnect=True` is a no-op for finite files (not a live source)."""
    factory_calls = []

    def factory(s):
        factory_calls.append(s)
        return FakeCapture([(True, _frame(0)), (False, None)])

    out = list(video_frames("clip.mp4", reconnect=True, capture_factory=factory))
    assert len(out) == 1
    assert len(factory_calls) == 1  # opened once, never re-opened


# --- reconnect with backoff (live) -------------------------------------------


def test_live_reconnects_after_read_failure(monkeypatch):
    """First capture yields a frame then fails; reconnect opens a second capture
    that yields more frames. drop_to_latest off to keep the test deterministic."""
    monkeypatch.setattr(time, "sleep", lambda _s: None)  # no real backoff wait

    caps = [
        FakeCapture([(True, _frame(1)), (False, None)]),
        FakeCapture([(True, _frame(2)), (True, _frame(3))]),
    ]
    made = []

    def factory(_s):
        cap = caps[len(made)]
        made.append(cap)
        return cap

    out = list(
        video_frames(
            "rtsp://cam/stream",
            drop_to_latest=False,
            max_frames=3,
            capture_factory=factory,
        )
    )
    assert len(out) == 3
    assert len(made) == 2  # reconnected exactly once
    assert caps[0].released  # old capture cleaned up on reconnect


def test_live_gives_up_after_max_attempts(monkeypatch):
    """Reconnect target never re-opens -> give up cleanly after the bound."""
    monkeypatch.setattr(time, "sleep", lambda _s: None)

    first = FakeCapture([(True, _frame(1)), (False, None)])
    made = []

    def factory(_s):
        if not made:
            made.append(first)
            return first
        made.append("dead")
        return FakeCapture([], opened=False)  # never opens

    out = list(
        video_frames(
            "rtsp://cam/stream",
            drop_to_latest=False,
            max_reconnect_attempts=3,
            capture_factory=factory,
        )
    )
    assert len(out) == 1  # only the one good frame before the drop
    # 1 initial open + 3 failed reconnect attempts
    assert len(made) == 1 + 3


def test_live_reconnect_disabled_ends_like_file(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    cap = FakeCapture([(True, _frame(1)), (False, None)])
    out = list(
        video_frames(
            "rtsp://cam/stream",
            reconnect=False,
            drop_to_latest=False,
            capture_factory=lambda s: cap,
        )
    )
    assert len(out) == 1
    assert cap.released


# --- drop-to-latest back-pressure --------------------------------------------


class FastCapture:
    """An effectively-infinite live producer: every read returns a fresh frame
    immediately (a tiny sleep yields the GIL so the consumer thread can run). It
    never reports EOF, so it models a live camera, not a finite file. Used to
    prove a SLOW consumer DROPS stale frames instead of queueing them."""

    def __init__(self, throttle=0.0005):
        self._i = 0
        self._throttle = throttle
        self._lock = threading.Lock()
        self.released = False

    def isOpened(self):  # noqa: N802
        return True

    def read(self):
        if self._throttle:
            time.sleep(self._throttle)  # pace the producer; yields the GIL
        with self._lock:
            self._i += 1
            return (True, _frame(self._i))

    @property
    def produced(self):
        with self._lock:
            return self._i

    def get(self, _prop):
        return 30.0

    def release(self):
        self.released = True


def test_drop_to_latest_skips_stale_frames():
    """With a fast producer and a slow consumer, the consumer must see far fewer
    frames than were produced (frames were dropped, not queued)."""
    cap = FastCapture()

    gen = video_frames(
        "rtsp://cam/stream",
        drop_to_latest=True,
        max_frames=5,
        capture_factory=lambda s: cap,
    )

    consumed = []
    for frame, ts in gen:
        consumed.append((frame, ts))
        time.sleep(0.05)  # slow consumer: 50ms per frame

    assert len(consumed) == 5
    # The producer ran far ahead while we slept; we consumed only the latest few.
    assert cap.produced > len(consumed) * 5
    # Wall-clock timestamps, monotonic non-decreasing, recent.
    ts = [t for _, t in consumed]
    assert all(b >= a for a, b in zip(ts, ts[1:], strict=False))
    assert ts[0] > time.time() - 10


def test_drop_to_latest_uses_wallclock_timestamps():
    cap = FastCapture()
    before = time.time()
    out = list(
        video_frames(
            "rtsp://cam/stream",
            drop_to_latest=True,
            max_frames=3,
            capture_factory=lambda s: cap,
        )
    )
    after = time.time()
    assert len(out) == 3
    for _, ts in out:
        assert before <= ts <= after
