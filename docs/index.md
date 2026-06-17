# Trio Retina

**Turn any perception model's output into one standard, queryable world-state — symbolic events + latent vectors.** The model-agnostic state layer for world models.

Retina turns raw signals — video, RTSP, files — into a queryable world-state: readable events (`zone.enter`, `dwell`, `line.cross`) plus optional latent vectors, on one small model-agnostic standard. Bring any model (YOLO, V-JEPA, DINO, a VLM, or none); Retina assembles its output into state a dynamics model, rule engine, or LLM can consume.

Think **OpenTelemetry for perception** — it doesn't build the sensors, it normalizes any of them into one state.

## Install

```bash
pip install trio-retina            # core: numpy only
pip install 'trio-retina[yolo]'    # + Ultralytics YOLO adapter
pip install 'trio-retina[video]'   # + OpenCV frame source (files / RTSP / webcam)
pip install 'trio-retina[all]'     # everything
```

## Quickstart

```python
from retina import Retina, Zone, ZoneRule, YoloDetector
from retina.sources import video_frames

dock = Zone("dock", [(0.3, 0.2), (0.7, 0.2), (0.7, 0.9), (0.3, 0.9)], normalized=True)

cam = Retina(
    source_id="cam_01",
    detector=YoloDetector("yolo11n.pt", classes={"person"}),
    rules=[ZoneRule(dock, classes={"person"}, dwell_s=30)],
)
for event in cam.run(video_frames("dock.mp4")):
    print(event.to_json())
```

No model, no GPU? Every demo runs on synthetic detections — start with `python examples/quickstart.py`.

## Where to go next

- **[Event spec](spec.md)** — the tiny, JWT-style `retina.event` interchange format.
- **[Design notes](design.md)** — why Retina is the *encoder* of a world model, and what it deliberately does not do.
- **[API reference](api.md)** — the public Python API, generated from docstrings.

For the full landing page (demos, supported models, comparisons, roadmap), see the [README on GitHub](https://github.com/machinefi/trio-retina#readme).
