"""The `retina` command-line interface (stdlib `argparse` only).

Kept dependency-light on purpose: the always-imported path is stdlib + numpy, so
`retina demo` runs the moment `pip install trio-retina` finishes — no model, no
GPU, no video. Anything heavy (OpenCV for `run` over a video/RTSP source) is
lazy-imported inside the subcommand that needs it, with a friendly pointer at the
right extra.

Subcommands:
  retina demo                       run a built-in synthetic demo, print events
  retina run <workflow.json> <src>  run a declarative pipeline over a source
  retina validate <events.jsonl>    validate a JSONL event stream
  retina bench                      Retina-layer overhead micro-benchmark
  retina --version                  print the package version
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from typing import TextIO

from . import __version__

# --- demo ---------------------------------------------------------------------


class _ScriptedDetector:
    """One 'person' box marching left-to-right, one step per call. Mirrors the
    synthetic detector in examples/quickstart.py so `retina demo` needs no model."""

    def __init__(self) -> None:
        self._xs = list(range(0, 102, 6))
        self._i = 0

    def __call__(self, frame):  # noqa: ANN001 - frame is an unused numpy array
        from .detect import Detection

        if self._i >= len(self._xs):
            return []
        x = self._xs[self._i]
        self._i += 1
        return [Detection(label="person", bbox=(x - 10, 40, x + 10, 60), confidence=0.9)]


def _demo_events() -> Iterator:
    """Yield the synthetic demo's events (numpy-only, no model/GPU/video)."""
    import numpy as np

    from .pipeline import Retina
    from .rules import CountRule, LineRule, ZoneRule
    from .track import IoUTracker
    from .zones import Line, Zone

    dock = Zone("dock", [(40, 0), (60, 0), (60, 100), (40, 100)])
    tripwire = Line("door", (50, 0), (50, 100))
    cam = Retina(
        source_id="cam_01",
        detector=_ScriptedDetector(),
        tracker=IoUTracker(min_hits=2),
        rules=[
            ZoneRule(dock, classes={"person"}, dwell_s=2.0),
            LineRule(tripwire, classes={"person"}),
            CountRule(threshold=1, classes={"person"}),
        ],
    )
    frames = [(np.zeros((100, 100, 3), dtype=np.uint8), float(i)) for i in range(18)]
    yield from cam.run(frames)


def _cmd_demo(args: argparse.Namespace, out: TextIO) -> int:
    n = 0
    for event in _demo_events():
        out.write(event.to_json() + "\n")
        n += 1
    if not args.quiet:
        print(f"retina demo: emitted {n} event(s) from the synthetic dock scene.", file=sys.stderr)
    return 0


# --- run ----------------------------------------------------------------------


def _source_frames(source: str) -> Iterable[tuple]:
    """Resolve a CLI source string to an iterable of (frame, timestamp) pairs.

    A video path / `rtsp://` URL goes through `retina.sources.video_frames`
    (OpenCV, lazy-imported). A bare integer string is treated as a webcam index."""
    src: str | int = int(source) if source.isdigit() else source
    try:
        from .sources import video_frames
    except ImportError as e:  # pragma: no cover - defensive
        raise SystemExit(
            "retina run needs OpenCV for video/RTSP sources. "
            "Install with: pip install 'trio-retina[video]'"
        ) from e
    return video_frames(src)


def _cmd_run(args: argparse.Namespace, out: TextIO) -> int:
    from .pipeline import Pipeline

    pipe = Pipeline.from_json(args.workflow)
    try:
        frames = _source_frames(args.source)
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 2

    sink = None
    if args.jsonl:
        from .export import JsonlSink

        sink = JsonlSink(args.jsonl)

    n = 0
    try:
        for event in pipe.run(frames):
            n += 1
            if sink is not None:
                sink(event)
            else:
                out.write(event.to_json() + "\n")
    finally:
        if sink is not None:
            sink.close()
    if args.jsonl:
        print(f"retina run: wrote {n} event(s) to {args.jsonl}", file=sys.stderr)
    return 0


# --- validate -----------------------------------------------------------------


def _cmd_validate(args: argparse.Namespace, out: TextIO) -> int:
    from .schema import validate

    valid = 0
    invalid = 0
    problems: list[tuple[int, str, list[str]]] = []

    try:
        fp = open(args.path)
    except OSError as e:
        print(f"retina validate: cannot open {args.path}: {e}", file=sys.stderr)
        return 2

    with fp:
        for lineno, raw in enumerate(fp, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                invalid += 1
                problems.append((lineno, line, [f"invalid JSON: {e.msg}"]))
                continue
            errs = validate(obj)
            if errs:
                invalid += 1
                problems.append((lineno, line, errs))
            else:
                valid += 1

    total = valid + invalid
    out.write(f"{total} event(s): {valid} valid, {invalid} invalid\n")
    if problems:
        shown = problems[: args.max_problems]
        out.write(f"first {len(shown)} problem(s):\n")
        for lineno, _line, errs in shown:
            out.write(f"  line {lineno}: {'; '.join(errs)}\n")
        if len(problems) > len(shown):
            out.write(f"  ... and {len(problems) - len(shown)} more\n")
    return 1 if invalid else 0


# --- bench --------------------------------------------------------------------


def _cmd_bench(args: argparse.Namespace, out: TextIO) -> int:
    import time

    import numpy as np

    from .detect import Detection
    from .pipeline import Retina
    from .rules import CountRule, LineRule, ZoneRule
    from .track import IoUTracker
    from .zones import Line, Zone

    w = h = 640
    n, k, warmup = args.frames, args.tracks, args.warmup

    def make_dets(frame_idx: int) -> list[Detection]:
        out_dets = []
        for j in range(k):
            cx = (frame_idx * 3 + j * 37) % w
            cy = (j * 53) % h
            out_dets.append(
                Detection(label="person", bbox=(cx - 8, cy - 8, cx + 8, cy + 8), confidence=0.9)
            )
        return out_dets

    precomputed = {i: make_dets(i) for i in range(warmup + n)}
    counter = {"i": 0}

    def detector_fn(_frame):
        i = counter["i"]
        counter["i"] += 1
        return precomputed[i]

    from .detect import CallableDetector

    zone = Zone("z", [(0.3, 0.0), (0.7, 0.0), (0.7, 1.0), (0.3, 1.0)], normalized=True)
    line = Line("door", (0.5, 0.0), (0.5, 1.0), normalized=True)
    cam = Retina(
        source_id="cam",
        detector=CallableDetector(detector_fn),
        tracker=IoUTracker(min_hits=1),
        rules=[
            ZoneRule(zone, classes={"person"}, dwell_s=1.0),
            LineRule(line, classes={"person"}),
            CountRule(threshold=1, classes={"person"}),
        ],
    )

    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(warmup):
        cam.process(img, float(i))

    t0 = time.perf_counter()
    for i in range(warmup, warmup + n):
        cam.process(img, float(i))
    full = time.perf_counter() - t0

    # Detector-only pass, to subtract the synthetic detector's trivial cost.
    counter2 = {"i": warmup}
    t0 = time.perf_counter()
    for _ in range(n):
        precomputed[counter2["i"]]
        counter2["i"] += 1
    det = time.perf_counter() - t0

    full_ms = full / n * 1e3
    det_ms = det / n * 1e3
    overhead_ms = (full - det) / n * 1e3

    out.write(f"frames={n}  tracks/frame={k}  (warmup={warmup})\n")
    out.write(f"  full pipeline  : {full_ms:.4f} ms/frame\n")
    out.write(f"  detector stub  : {det_ms:.4f} ms/frame  (excluded)\n")
    out.write(f"  Retina overhead: {overhead_ms:.4f} ms/frame at {k} tracks\n")
    return 0


# --- parser -------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retina",
        description="Turn camera streams into event streams — CLI.",
    )
    parser.add_argument("--version", action="version", version=f"retina {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p_demo = sub.add_parser(
        "demo", help="run a built-in synthetic demo (numpy-only) and print the event stream"
    )
    p_demo.add_argument("-q", "--quiet", action="store_true", help="suppress the summary line")
    p_demo.set_defaults(func=_cmd_demo)

    p_run = sub.add_parser(
        "run", help="run a declarative workflow (JSON) over a video / RTSP source"
    )
    p_run.add_argument("workflow", help="path to a workflow.json (Pipeline.from_json)")
    p_run.add_argument("source", help="video file path, rtsp:// URL, or webcam index")
    p_run.add_argument("--jsonl", metavar="OUT", help="write events to a JSONL file instead of stdout")
    p_run.set_defaults(func=_cmd_run)

    p_val = sub.add_parser("validate", help="validate a JSONL event stream against retina.event/0.1")
    p_val.add_argument("path", help="path to an events.jsonl file")
    p_val.add_argument(
        "--max-problems", type=int, default=10, help="max problems to list (default: 10)"
    )
    p_val.set_defaults(func=_cmd_validate)

    p_bench = sub.add_parser("bench", help="Retina-layer overhead micro-benchmark (ms/frame)")
    p_bench.add_argument("--frames", type=int, default=2000, help="frames to time (default: 2000)")
    p_bench.add_argument("--tracks", type=int, default=20, help="objects per frame (default: 20)")
    p_bench.add_argument("--warmup", type=int, default=200, help="warmup frames (default: 200)")
    p_bench.set_defaults(func=_cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args, sys.stdout)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
