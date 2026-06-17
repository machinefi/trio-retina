# Trio Retina

**Turn any perception model's output into one standard, queryable world-state — symbolic events, with a latent-vector channel built in.** The model-agnostic state layer for world models.

Retina turns raw signals — video, RTSP, files — into a queryable world-state: readable events (`zone.enter`, `dwell`, `line.cross`) *plus* a standardized **latent** `vec` channel on the same records, on one small model-agnostic standard. The latent channel is a real, serializable interface today (attach your own embedding — see [`examples/latent_vec.py`](https://github.com/machinefi/trio-retina/blob/main/examples/latent_vec.py)); the automatic *producers* (V-JEPA scene + per-object ReID) are on the [roadmap](https://github.com/machinefi/trio-retina#-roadmap). Bring any model (YOLO, V-JEPA, DINO, a VLM, or none); Retina assembles its output into state a dynamics model, rule engine, or LLM can consume.

Think **OpenTelemetry for perception** — it doesn't build the sensors, it normalizes any of them into one state.

## Install

```bash
pip install trio-retina            # core: numpy only
pip install 'trio-retina[yolo]'    # + Ultralytics YOLO adapter
pip install 'trio-retina[video]'   # + OpenCV frame source (files / RTSP / webcam)
pip install 'trio-retina[all]'     # everything
```

## Quickstart

Runs on a bare `pip install trio-retina` (numpy only) — no model, no GPU, no video file. A stand-in detector walks one "person" across a dock zone; Retina emits the real `retina.event` stream:

```python
import numpy as np

from retina import CountRule, IoUTracker, Retina, Zone, ZoneRule
from retina.detect import Detection


class ScriptedDetector:
    """A stand-in model: one 'person' walking across a dock zone."""

    def __init__(self):
        self._xs = list(range(0, 102, 6))

    def __call__(self, frame):
        x = self._xs.pop(0) if self._xs else 100
        return [Detection(label="person", bbox=(x - 10, 40, x + 10, 60), confidence=0.9)]


dock = Zone("dock", [(40, 0), (60, 0), (60, 100), (40, 100)])

cam = Retina(
    source_id="cam_01",
    detector=ScriptedDetector(),
    tracker=IoUTracker(min_hits=2),
    rules=[
        ZoneRule(dock, classes={"person"}, dwell_s=2.0),
        CountRule(1, classes={"person"}),
    ],
)

frames = [(np.zeros((100, 100, 3), dtype=np.uint8), float(i)) for i in range(18)]
for event in cam.run(frames):
    print(event.to_json())
    # {"type":"count.threshold","t":1.0,"src":"cam_01","n":1,"frame":1,...}
    # {"type":"zone.enter","t":7.0,"src":"cam_01","id":1,"label":"person",...}
    # {"type":"zone.dwell","t":7.0,...,"zone":"dock","dur":2.0,...}
    # {"type":"zone.exit","t":7.0,...,"zone":"dock","dur":3.0,...}
```

### With a real model + video

`pip install 'trio-retina[yolo]'` (add `[video]` for the frame source), then point it at your clip:

```python
from retina import Retina, Zone, ZoneRule, YoloDetector
from retina.sources import video_frames

dock = Zone("dock", [(0.3, 0.2), (0.7, 0.2), (0.7, 0.9), (0.3, 0.9)], normalized=True)

cam = Retina(
    source_id="cam_01",
    detector=YoloDetector("yolo11n.pt", classes={"person"}),
    rules=[ZoneRule(dock, classes={"person"}, dwell_s=30)],
)
for event in cam.run(video_frames("your.mp4")):
    print(event.to_json())
```

### Run it in your browser — no install

| notebook | what it shows |
|---|---|
| [quickstart](https://colab.research.google.com/github/machinefi/trio-retina/blob/main/notebooks/01_quickstart_events.ipynb) | detector → `zone` / `line` / `count` / `dwell` events + `validate()` |
| [camera → webhook](https://colab.research.google.com/github/machinefi/trio-retina/blob/main/notebooks/02_camera_to_webhook.ipynb) | a restricted-zone alert pushed to your endpoint |
| [from Supervision](https://colab.research.google.com/github/machinefi/trio-retina/blob/main/notebooks/03_from_supervision.ipynb) | pipe your existing `sv.Detections` straight in |

More no-model examples ship with the source (not the wheel) — `git clone` the repo and run `python examples/quickstart.py`. See also [`rtsp_to_webhook.py`](https://github.com/machinefi/trio-retina/blob/main/examples/rtsp_to_webhook.py), [`from_supervision.py`](https://github.com/machinefi/trio-retina/blob/main/examples/from_supervision.py), and [`latent_vec.py`](https://github.com/machinefi/trio-retina/blob/main/examples/latent_vec.py).

## Where to go next

- **[Event spec](spec.md)** — the tiny, JWT-style `retina.event` interchange format.
- **[Design notes](design.md)** — why Retina is the *encoder* of a world model, and what it deliberately does not do.
- **[API reference](api.md)** — the public Python API, generated from docstrings.

For the full landing page (demos, supported models, comparisons, roadmap), see the [README on GitHub](https://github.com/machinefi/trio-retina#readme).
