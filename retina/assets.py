"""Sample assets so the examples / docs / CLI run out of the box.

Two tiny helpers, both exported from the top-level package:

- `sample_events()` — a path to a small **bundled** `retina.event` JSONL that
  ships inside the wheel. Zero network, zero licensing risk: you can validate it,
  feed it to the CLI, or read it as a worked example of the event format the
  moment `pip install trio-retina` finishes.
- `sample_video()` — a path to a small clip for exercising the *video-source*
  plumbing (`video_frames`, `retina run`, a real `YoloDetector`). To stay clear
  of third-party-footage licensing entirely, the clip is **generated
  synthetically** on first call (deterministic moving shapes) and cached in a
  per-user dir. It is *not* real-world footage — see `sample_video` for what that
  does and does not buy you.

Both are stdlib-only on the always-imported path (no torch, no OpenCV at import);
`sample_video()` lazily needs `[video]` only to *write* the synthetic clip.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

__all__ = ["sample_events", "sample_video"]


def sample_events() -> str:
    """Return a filesystem path to the bundled sample `retina.event` JSONL.

    The file ships inside the package (``retina/_assets/sample_events.jsonl``),
    so this works **offline** the instant the wheel is installed — no network,
    no licensing risk. It is the five-event synthetic dock scene from
    ``retina demo`` (count threshold → zone enter → dwell → line cross → exit),
    handy for trying the event format, ``validate()``, or the CLI::

        retina validate "$(python -c 'import retina; print(retina.sample_events())')"

    Returns the path as a ``str``. The path is stable for the life of the
    process; treat the file as read-only (it lives inside the install).
    """
    # `resources.files(...)` returns a Traversable; for a regular wheel install
    # it is already a real path. `as_file` would copy from a zip, but we ship an
    # unzipped wheel layout, so the direct path is correct and cheap.
    res = resources.files("retina").joinpath("_assets", "sample_events.jsonl")
    path = Path(str(res))
    if not path.is_file():  # pragma: no cover - defensive; asset ships in the wheel
        raise FileNotFoundError(
            f"bundled sample events not found at {path!r}. This usually means a "
            "broken install — reinstall with: pip install --force-reinstall trio-retina"
        )
    return str(path)


def _cache_dir() -> Path:
    """Per-user cache dir for downloaded/generated sample assets.

    Honors ``XDG_CACHE_HOME``; falls back to ``~/.cache``. Created on demand."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(Path.home(), ".cache")
    d = Path(base) / "trio-retina"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Synthetic clip parameters — deterministic, so the cached file is reproducible.
_SYNTH_NAME = "sample_synthetic_640x360.mp4"
_SYNTH_W, _SYNTH_H, _SYNTH_FPS, _SYNTH_FRAMES = 640, 360, 20, 80


def sample_video(*, force: bool = False) -> str:
    """Return a path to a small sample video clip, cached per-user.

    **What this is.** A *synthetic* clip — deterministic moving shapes (a couple
    of coloured rectangles drifting across a dark background) — generated once
    with OpenCV and cached under ``~/.cache/trio-retina/``. It exists to exercise
    the **video-source plumbing** end to end with zero network and zero
    third-party-footage licensing risk: ``video_frames(retina.sample_video())``,
    ``retina run workflow.json "$(... sample_video ...)"``, frame striding, EOF
    handling, and so on.

    **What this is NOT.** It is not real-world footage, so a real object detector
    (``YoloDetector``) will find no people/vehicles in it — there are none. For
    the YOLO-on-real-footage path, point Retina at **your own clip**
    (``video_frames("your.mp4")``); the synthetic clip only verifies the wiring.

    Writing the clip needs OpenCV (the ``[video]`` extra). The first call writes
    and caches it; later calls return the cached path immediately. Pass
    ``force=True`` to regenerate.

    Raises ``RuntimeError`` with a clear ``[video]`` hint if OpenCV is missing
    and the clip is not already cached.
    """
    path = _cache_dir() / _SYNTH_NAME
    if path.is_file() and not force:
        return str(path)
    _write_synthetic_clip(path)
    return str(path)


def _write_synthetic_clip(path: Path) -> None:
    """Write the deterministic synthetic clip to ``path`` via OpenCV.

    Two coloured rectangles drift across a dark frame at constant velocity —
    enough motion to drive a ``MotionGate``, a tracker, and the source plumbing,
    while staying tiny and reproducible. No randomness: byte-identical each run.
    """
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError(
            "sample_video() needs OpenCV to write the synthetic clip. "
            "Install with: pip install 'trio-retina[video]' "
            "(or supply your own clip path to video_frames(...))."
        ) from e

    import numpy as np

    w, h, fps, n = _SYNTH_W, _SYNTH_H, _SYNTH_FPS, _SYNTH_FRAMES
    # OpenCV picks the container from the file *extension*, so the temp file must
    # keep `path`'s suffix (e.g. `foo.mp4.<pid>` would not open). Stage as a
    # sibling hidden dotfile that preserves the real extension, then atomically
    # rename onto `path` so a crashed write never leaves a half-clip in the cache.
    tmp = path.with_name(f".{path.stem}.{os.getpid()}{path.suffix}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp), fourcc, fps, (w, h))
    if not writer.isOpened():  # pragma: no cover - codec/platform dependent
        raise RuntimeError(
            f"OpenCV could not open a VideoWriter for {tmp!r} (mp4v codec "
            "unavailable on this platform). Supply your own clip to video_frames(...)."
        )
    try:
        for i in range(n):
            frame = np.full((h, w, 3), 24, dtype=np.uint8)  # dark gray background
            # Box A: drifts left->right; Box B: drifts top->bottom. Wrap around.
            ax = int((40 + i * 7) % (w - 60))
            by = int((30 + i * 4) % (h - 60))
            cv2.rectangle(frame, (ax, 150), (ax + 60, 210), (60, 180, 250), -1)
            cv2.rectangle(frame, (300, by), (360, by + 60), (250, 120, 60), -1)
            writer.write(frame)
    finally:
        writer.release()
    os.replace(tmp, path)  # atomic publish — partial writes never become the cache
