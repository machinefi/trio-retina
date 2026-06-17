# Trio Retina

**Turn any perception model's output into one standard, queryable world-state — symbolic events + latent vectors.**
The model-agnostic state layer for world models.

*A lightweight, model-agnostic **computer-vision pipeline** for **object detection & tracking** that emits structured **events** — zone intrusion, line-crossing, dwell, people-counting — from **YOLO**, **VLM**, or **Grounding DINO** detectors over video, files, or **RTSP**. Runs on CPU at the **edge**; feeds **digital twins**, dynamics models, and LLMs.*

[![CI](https://github.com/machinefi/trio-retina/actions/workflows/ci.yml/badge.svg)](https://github.com/machinefi/trio-retina/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

![Trio Retina computer-vision pipeline: YOLO object tracking with two dynamics models forecasting entity trajectories from one world-state](https://raw.githubusercontent.com/machinefi/trio-retina/main/media/retina_demo.gif)

> One world-state from any detector → **two dynamics models forecast where each entity is headed** off the *same* state (gray = constant-velocity, magenta = learned, −35%). Swap the detector (YOLO → V-JEPA → DINO) or the dynamics model — the state in the middle is the constant.

## 👋 hello

**Trio Retina** (Retina for short) turns raw signals — video, sensor — into a **queryable world-state**: readable **events** (`zone.enter`, `dwell`, `line.cross`) *plus* optional **latent** vectors, on one small model-agnostic standard. Bring any model (YOLO, V-JEPA, DINO, a VLM, or none); Retina assembles its output into state a dynamics model, rule engine, or LLM can consume.

Think **OpenTelemetry for perception** — it doesn't build the sensors, it normalizes any of them into one state. In world-model terms it's the **encoder** (`s = Enc(x)`), and *only* the encoder; dynamics and policy build on top. → see [`DESIGN.md`](DESIGN.md).

## 💻 install

From source — a PyPI `retina-sdk` release is landing shortly:

```bash
pip install "retina-sdk @ git+https://github.com/machinefi/trio-retina"          # core: numpy only
pip install "retina-sdk[yolo]  @ git+https://github.com/machinefi/trio-retina"   # + Ultralytics YOLO adapter
pip install "retina-sdk[video] @ git+https://github.com/machinefi/trio-retina"   # + OpenCV frame source (files / RTSP / webcam)
pip install "retina-sdk[all]   @ git+https://github.com/machinefi/trio-retina"   # everything
```

## 🔥 quickstart

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
    # {"type":"zone.dwell","t":1718254799.8,"src":"cam_01","id":42,
    #  "label":"person","zone":"dock","dur":31.0,"conf":0.91}
```

No model, no GPU? Every demo under [`examples/`](examples/) runs on synthetic detections — start with `python examples/quickstart.py`.

### compose models with `|`

Wire models like n8n / LangChain, no GUI. Add a cheap gate and a VLM enricher anywhere in the chain:

```python
from retina import MotionGate, GateNode, YoloDetector, IoUTracker, EnricherNode, ZoneRule, JsonlSink

pipe = (
    GateNode(MotionGate())                 # skip static frames (cut model calls)
    | YoloDetector("yolo11n.pt", classes={"person", "forklift"})
    | IoUTracker()
    | EnricherNode(my_vlm_describe)        # attach a VLM read to frame.user
    | ZoneRule(dock, dwell_s=30)
    | JsonlSink("events.jsonl")
)
```

<details>
<summary>Two more ways to wire it (explicit list · declarative JSON) + the node catalog</summary>

```python
# explicit node list
from retina import Pipeline, DetectorNode, TrackerNode, RuleNode
pipe = Pipeline([DetectorNode(yolo), TrackerNode(), RuleNode(ZoneRule(dock))])

# declarative workflow file (shareable, no code)
pipe = Pipeline.from_json("workflow.json")   # see examples/workflow.json
```

| node | what it does | wraps |
|---|---|---|
| `DetectorNode` | image → detections | any `callable(image)->[Detection]` |
| `TrackerNode` | detections → tracks | `IoUTracker` / `NorfairTracker` |
| `RuleNode` | tracks → events | `ZoneRule` / `LineRule` / `CountRule` |
| `GateNode` | drop uninteresting frames | any `callable(image,t)->bool` (e.g. `MotionGate`) |
| `EnricherNode` | attach context to `frame.user` | any `callable(frame)->dict` (VLM / V-JEPA) |
| `SinkNode` | emit events | `JsonlSink` / `WebhookSink` |

Register your own for `from_json` with `register_node("my_type", builder)`.
</details>

## 🎛️ supported models

Retina imports no model — **any** `callable(image) -> [Detection]` plugs in (`CallableDetector` wraps a function in one line). Batteries-included:

- **YOLO family** — `YoloDetector("<weights>.pt")` (Ultralytics): YOLOv5/8/9/10/11/12, RT-DETR. Open-vocab via YOLO-World.
- **Open-vocab from text** — `GroundingDinoDetector(["forklift", "hard hat"])`, no training.
- **Any VLM** — `VlmDetector(client, prompt)` (Qwen-VL / Gemini / GPT-4o / Claude / local), as a detector or an event-source enricher.
- **Supervision interop** — `Detection.from_supervision(sv_detections)` ingests a Roboflow `sv.Detections`, so anything that already converts to Supervision pipes straight into Retina's event layer.

Trackers are pluggable too: `IoUTracker` (pure-Python default) or `NorfairTracker`.

## 📦 the event format

The `retina.event` standard is tiny, like a JWT — three required fields, everything else optional and omitted when absent. Full spec in [`SPEC.md`](SPEC.md).

```json
{"type":"zone.dwell","t":1718254799.8,"src":"cam_01","id":42,"label":"person","zone":"dock","dur":31.0}
```

```python
from retina import validate
validate(event)   # -> [] if valid, else a list of problems  (pure-Python, ships a JSON Schema)
```

## 🎬 demos

### Forecast — the dynamics layer on top of Retina

The hero GIF above. [`examples/forecast/`](examples/forecast/) runs a dynamics model on Retina's `WorldState` stream and shows *why* Retina is necessary: a dynamics model eats structured **state**, not pixels.

### iTwin.js — a live, predictive layer for a digital twin

![Trio Retina perception events and forecast arrows rendered live on a Bentley iTwin.js digital twin (Baytown plant)](https://raw.githubusercontent.com/machinefi/trio-retina/main/examples/itwin/media/retina_itwin_demo.gif)

[`examples/itwin/`](examples/itwin/) drops Retina's entities, forecast arrows, and `retina.event` alerts onto a real Bentley **iTwin.js** iModel (the Baytown sample plant), through one neutral JSON contract — rendered fully headless. Retina doesn't replace the twin; it gives it *live eyes*.

<details>
<summary>All examples (each runs with no model / no GPU)</summary>

```bash
python examples/quickstart.py        # zone / line / count / dwell events
python examples/three_apps.py        # one stream -> security, retail, safety
python examples/any_model.py         # swap the detector, rest unchanged
python examples/gate_savings.py      # a cheap gate cuts detector calls 100 -> 23
python examples/pipeline_compose.py  # compose with | (n8n without a GUI)
python examples/yolo_video.py v.mp4  # real footage  (pip install 'retina-sdk[all]')
```
</details>

## 🎯 use cases

One state layer, many domains — the *same* `retina.event` stream, read differently above:

- **Security & intrusion detection** — `zone.enter` / `line.cross` on cameras and RTSP feeds.
- **Retail analytics & people-counting** — footfall, queue dwell, zone occupancy from any detector.
- **Workplace safety** — PPE, forklift, and restricted-zone alerts via open-vocab detectors.
- **Smart city & traffic monitoring** — vehicle/pedestrian counting and crossings at the edge.
- **Industrial digital twins** — feed live entities + forecasts into a twin ([iTwin.js demo](examples/itwin/)).

## 🧠 how it works

Everything flows through one append-only data unit, the **`Frame`**. Each stage *enriches* it and never overwrites upstream fields:

```
                      ┌──────────────── Frame (append-only) ───────────────┐
 frame ─► Detector ─► │ .detections ─► Tracker ─► .tracks ─► Rule ─► .events │ ─► Sink
   ▲        ▲         │                  ▲                    ▲              │     ▲
 source   any model   │   Gate (skip?)   tracker     zone/line/count/dwell  │  jsonl/
                      │   Enricher (VLM / V-JEPA → .user)                    │  webhook
                      └─────────────────────────────────────────────────────┘
```

- The **detector** is the model-agnostic seam: any `callable(image) -> [Detection]`.
- The **tracker** gives objects identity over time; **rules** turn tracks into **events**; **enrichers** attach context; **gates** skip work; **sinks** push out.
- Output is **dual**: a readable symbolic stream *and* an optional model-tagged latent channel — never collapsed.

<details>
<summary>Why "encoder", the dual state, and how it compares to DeepStream / Supervision</summary>

**Two senses of "encoder."** Foundation backbones (V-JEPA, DINO, SAM, YOLO) turn pixels into features — that race is theirs, and Retina rides it. Retina is the encoder *layer* on top: it **fuses** many models into one record, gives objects **persistent identity**, **structures** it into entities + relations + events, carries the **dual** symbolic + latent channels, as an **event-sourced stream** — one small, serializable, model-agnostic standard.

**Dual state.** The same entities on two linked channels: *symbolic* (readable `events` / entity records, for rules / LLMs / dashboards) and *latent* (optional model-tagged embeddings, for a downstream dynamics model). Symbols you can read; vectors a model can predict on.

**vs DeepStream / Holoscan** — same good ideas (event semantics, metadata model, composable graph), none of the weight:

| | DeepStream / Holoscan | **Retina** |
|---|---|---|
| Install | CUDA + TensorRT + containers | `pip install retina-sdk` |
| Hardware | NVIDIA / Jetson locked | any machine — CPU is fine |
| Model | tied to the NV stack | **bring any model** (or none) |
| Shape | a platform you build *inside* | a library you `import` |
| Core deps | a lot | **numpy** |

**vs Supervision** — Supervision turns a model's output into detections + overlays (great toolbox, ends at the screen). Retina is a level up: it emits a serializable **state + event stream** that the *next* layer (dynamics, twin, agent) consumes. We compose Supervision / detectors, not compete with them.

Full rationale, references, and the world-model stack: [`DESIGN.md`](DESIGN.md).
</details>

## 🗺️ roadmap

Early but real (`v0.1.0`). Stable: the event layer + JSON Schema/validator, the composable pipeline (`|` / list / JSON), YOLO + open-vocab + VLM detectors, IoU + Norfair trackers.

Next: ByteTrack / OC-SORT · `proximity` / `anomaly` events · VLM-as-event-source · Kafka / MQTT sinks · the **latent channel** (surface V-JEPA scene + per-object embeddings). See [`CHANGELOG.md`](CHANGELOG.md).

Retina is the open **perception encoder** extracted from [Trio](https://machinefi.com); the layers above (dynamics, policy / judgment) are Trio's commercial platform. Retina is, and stays, model-agnostic and free.

## 🤝 contributing

Contributions that keep Retina small and beautiful are very welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md) for dev setup and how to add a detector / tracker / rule / sink. By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md); to report a vulnerability see [`SECURITY.md`](SECURITY.md).

## license

[Apache-2.0](LICENSE).
