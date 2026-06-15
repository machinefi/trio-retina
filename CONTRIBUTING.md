# Contributing to Retina

Thanks for your interest! Retina is a small, model-agnostic perception/state
layer. Contributions that keep it small and beautiful are very welcome.

## Dev setup

Retina has a numpy-only core, so a basic environment is light:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
```

## Tests & lint

```bash
pytest -q             # run the test suite
ruff check .          # lint
ruff format .         # auto-format (line-length = 100)
```

Both `ruff check .` and `pytest -q` must be green before you open a PR — CI runs
exactly these across Python 3.10–3.13.

## Design principles

Please keep changes aligned with these:

- **Small & beautiful.** Prefer a few sharp, composable primitives over broad
  surface area. New abstractions need to earn their keep.
- **Pure-Python core, numpy-only.** Everything in `retina/` that's always
  importable must depend on nothing heavier than `numpy`.
- **Model / hardware / app-agnostic.** The core never assumes a specific
  detector, tracker, camera, accelerator, or downstream application.
- **Heavy deps are optional extras.** Anything that pulls in a model runtime or
  large library lives behind an extra and is imported lazily:
  `[yolo]` (ultralytics), `[norfair]`, `[grounding]` (transformers/torch),
  `[video]` (opencv). Import the optional package *inside* the class that needs
  it, with a friendly error if it's missing.

## Extending Retina

The building blocks are tiny `Protocol`s in `retina/` — implement one and it
drops straight into a pipeline. Anything pipeable composes with `|` (see
`retina/compose.py` and `retina/nodes.py`):

```python
pipe = YoloDetector("yolo11n.pt") | IoUTracker() | ZoneRule(dock) | JsonlSink("e.jsonl")
```

- **Detector** (`retina/detect.py`) — `__call__(self, frame: np.ndarray) -> list[Detection]`.
  Turns an image into detections. See `YoloDetector`, `VlmDetector`.
- **Tracker** (`retina/track.py`) — `update(self, detections, t) -> list[Track]`.
  Gives detections identity over time. See `IoUTracker`, `NorfairTracker`.
- **EventRule** (`retina/rules.py`) — `update(self, tracks, t, frame_idx) -> list[Event]`.
  Emits symbolic events from tracks. See `ZoneRule`, `LineRule`, `CountRule`.
- **EventSink** (`retina/export.py`) — `__call__(self, event: Event) -> None`.
  Consumes events. See `JsonlSink`, `WebhookSink`.

To wrap a plain function or a custom object, see the `Node` types in
`retina/nodes.py`; shipped detectors/trackers/rules/sinks auto-wrap, so you
rarely need an explicit node.

## Pull requests

- Keep PRs focused; one concern per PR.
- Add or update tests for behavior you change.
- Make sure `ruff check .` and `pytest -q` pass locally.
- Describe the *why*, not just the *what*, in the PR description.

By contributing you agree your work is licensed under the project's
[Apache-2.0](LICENSE) license.
