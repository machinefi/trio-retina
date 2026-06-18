"""Frame sources.

A source is just an iterable of `(frame, timestamp)` pairs, so the core never
depends on OpenCV — you can feed it a list of numpy arrays in tests. The
`video_frames` helper is an optional convenience that uses OpenCV (install with
`pip install 'trio-retina[video]'`) for files / RTSP / webcam.

Edge / robotics notes
---------------------
Live sources (RTSP, webcam, or any source opened with ``live=True``) are hardened
for unattended deployment:

* **Reconnect with backoff.** A live ``cap.read()`` failure is treated as a
  transient drop, not end-of-stream: the capture is re-opened with exponential
  backoff (``reconnect_initial`` → ``reconnect_max`` seconds, capped by
  ``max_reconnect_attempts`` / ``reconnect_timeout``) and the generator resumes.
  A real file, by contrast, ends its generator at EOF exactly as before — finite
  media is never reconnected.

* **Drop-to-latest back-pressure.** When the consumer (detector + rules) is
  slower than the camera, a live feed must not buffer an ever-growing backlog of
  stale frames. A small background reader keeps only the *newest* frame in a
  one-slot handoff, so a slow consumer always sees fresh frames and latency stays
  bounded. Wall-clock timestamps are taken in the reader thread, as close to the
  grab as possible. Finite files never drop frames (every frame is delivered).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator

import numpy as np

logger = logging.getLogger("retina.sources")


def _is_live_source(source: str | int, live: bool) -> bool:
    """A source is 'live' when explicitly flagged, an integer webcam index, or a
    streaming URL (rtsp/rtmp/udp/http-stream). Plain file paths are NOT live."""
    if live:
        return True
    if isinstance(source, int):
        return True
    if isinstance(source, str):
        lowered = source.lower()
        return lowered.startswith(("rtsp://", "rtmp://", "udp://", "rtp://"))
    return False


def _default_capture_factory(source: str | int):
    """Open an OpenCV VideoCapture. Imported lazily so numpy-only installs (and
    tests, which inject a fake factory) never need cv2."""
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - exercised only with extra
        raise ImportError(
            "video_frames needs OpenCV. Install with: pip install 'trio-retina[video]'"
        ) from e
    return cv2.VideoCapture(source)


def _try_set_buffersize_1(cap) -> None:
    """Best-effort `CAP_PROP_BUFFERSIZE = 1` so the OpenCV/FFmpeg layer keeps the
    smallest possible backlog. Silently ignored if cv2/driver doesn't support it."""
    try:
        import cv2

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:  # pragma: no cover - driver/codec dependent
        pass


class _LatestFrameReader:
    """Threaded reader that keeps only the newest (frame, wall-clock ts) in a
    one-slot handoff — drop-to-latest back-pressure for live sources.

    The reader thread grabs as fast as the source produces; a slow consumer
    calling `read()` always gets the freshest frame and never a stale backlog.
    On a read failure the thread stops and surfaces the failure to the consumer
    (which then drives reconnection at the generator level).
    """

    def __init__(self, cap):
        self._cap = cap
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest: tuple[np.ndarray, float] | None = None
        self._failed = False
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            with self._lock:
                if self._stop:
                    return
            ok, frame = self._cap.read()
            ts = time.time()  # stamp as close to grab as possible
            with self._cond:
                if self._stop:
                    return
                if not ok:
                    self._failed = True
                    self._cond.notify_all()
                    return
                self._latest = (frame, ts)
                self._cond.notify_all()

    def read(self, timeout: float = 1.0) -> tuple[bool, np.ndarray | None, float]:
        """Block until a fresh frame is available, the reader failed, or timeout.

        Returns `(ok, frame, ts)`. `ok=False` means the underlying read failed
        (caller should reconnect); `frame=None` with `ok=True` never happens."""
        deadline = time.time() + timeout
        with self._cond:
            while True:
                if self._failed:
                    return False, None, 0.0
                if self._latest is not None:
                    frame, ts = self._latest
                    self._latest = None  # consume; next read waits for a newer one
                    return True, frame, ts
                remaining = deadline - time.time()
                if remaining <= 0:
                    # No new frame in time, but the source is still alive. Signal
                    # "ok, nothing new" so the generator can loop without dropping.
                    return True, None, 0.0
                self._cond.wait(timeout=remaining)

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self._thread.join(timeout=1.0)


def video_frames(
    source: str | int,
    *,
    stride: int = 1,
    live: bool = False,
    max_frames: int | None = None,
    reconnect: bool = True,
    max_reconnect_attempts: int | None = None,
    reconnect_timeout: float | None = None,
    reconnect_initial: float = 0.5,
    reconnect_max: float = 30.0,
    drop_to_latest: bool | None = None,
    capture_factory=None,
) -> Iterator[tuple[np.ndarray, float]]:
    """Yield (frame, timestamp) from a file path, RTSP/HLS URL, or webcam index.

    `stride` samples every Nth frame (cheap frame-rate reduction). Timestamps are
    media-time (frame_idx / fps) for files, or wall-clock epoch when the source is
    live (`live=True`, an integer webcam index, or an `rtsp://`/`rtmp://`/`udp://`
    URL).

    Live-source hardening (no effect on finite files — a real EOF still ends the
    generator):

    * `reconnect` (default True): on a live read failure, re-open the capture with
      exponential backoff (`reconnect_initial` → `reconnect_max` seconds) and
      resume instead of ending. Bounded by `max_reconnect_attempts` (consecutive
      attempts) and/or `reconnect_timeout` (wall-clock seconds); give up cleanly
      when either is exceeded.
    * `drop_to_latest` (default True for live): run a background reader that keeps
      only the newest frame, so a slow consumer gets bounded latency instead of a
      growing backlog. Also sets `CAP_PROP_BUFFERSIZE = 1` best-effort.

    `capture_factory` is an injection seam: a callable `source -> capture` where
    `capture` has `.isOpened()`, `.read() -> (ok, frame)`, `.get(prop)`, and
    `.release()`. Tests pass a fake here so no real camera/cv2 is needed.
    """
    is_live = _is_live_source(source, live)
    factory = capture_factory or _default_capture_factory
    if drop_to_latest is None:
        drop_to_latest = is_live
    use_threaded = is_live and drop_to_latest

    def _open():
        cap = factory(source)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    cap = _open()
    if cap is None:
        raise RuntimeError(f"could not open video source: {source!r}")
    if use_threaded:
        _try_set_buffersize_1(cap)

    fps = cap.get(cv_prop_fps()) or 30.0
    idx = 0
    emitted = 0
    reader = _LatestFrameReader(cap) if use_threaded else None
    deadline = (
        time.time() + reconnect_timeout
        if (is_live and reconnect and reconnect_timeout is not None)
        else None
    )

    def _reconnect() -> bool:
        """Re-open a live capture with exponential backoff. Returns True on
        success, False once attempt/timeout bounds are exhausted."""
        nonlocal cap, reader
        if reader is not None:
            reader.stop()
            reader = None
        if cap is not None:
            cap.release()
            cap = None
        backoff = reconnect_initial
        attempt = 0
        while True:
            attempt += 1
            if max_reconnect_attempts is not None and attempt > max_reconnect_attempts:
                logger.error(
                    "video_frames: giving up reconnecting to %r after %d attempts",
                    source,
                    attempt - 1,
                )
                return False
            if deadline is not None and time.time() >= deadline:
                logger.error(
                    "video_frames: giving up reconnecting to %r after timeout", source
                )
                return False
            logger.warning(
                "video_frames: live source %r dropped; reconnect attempt %d in %.1fs",
                source,
                attempt,
                backoff,
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, reconnect_max)
            new_cap = _open()
            if new_cap is not None:
                cap = new_cap
                if use_threaded:
                    _try_set_buffersize_1(cap)
                    reader = _LatestFrameReader(cap)
                logger.warning("video_frames: reconnected to %r", source)
                return True

    try:
        while True:
            if reader is not None:
                ok, frame, ts = reader.read()
                if ok and frame is None:
                    continue  # alive but no new frame yet — loop, don't drop/stop
            else:
                ok, frame = cap.read()
                ts = time.time() if is_live else idx / fps

            if not ok:
                if is_live and reconnect:
                    if _reconnect():
                        continue
                # finite file EOF, or reconnection exhausted/disabled -> end.
                break

            if idx % stride == 0 and frame is not None:
                yield frame, ts
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break
            idx += 1
    finally:
        if reader is not None:
            reader.stop()
        if cap is not None:
            cap.release()


def cv_prop_fps() -> int:
    """`cv2.CAP_PROP_FPS` without importing cv2 unless available. The constant is
    a stable OpenCV enum value (5); falls back to it so a fake capture's `.get(5)`
    works in numpy-only tests."""
    try:
        import cv2

        return cv2.CAP_PROP_FPS
    except ImportError:
        return 5
