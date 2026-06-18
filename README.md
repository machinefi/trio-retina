<div align="center">

# Trio Retina

<img src="https://raw.githubusercontent.com/machinefi/trio-retina/main/media/stack.png" width="840" alt="The world-model stack: perception backbones (YOLO, DINOv2, V-JEPA 2, SAM, VLMs) feed Trio Retina — the encoder and standardized WorldState — which world models for dynamics and control build on top of">

**The state layer of the world-model stack** — bring any perception model on top, get one standard, model-agnostic `WorldState`, build any dynamics underneath. Swap the model or the dynamics; **Retina is the constant in the middle.**

[![CI](https://github.com/machinefi/trio-retina/actions/workflows/ci.yml/badge.svg)](https://github.com/machinefi/trio-retina/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/trio-retina.svg)](https://pypi.org/project/trio-retina/) [![Docs](https://img.shields.io/badge/docs-live-brightgreen.svg)](https://machinefi.github.io/trio-retina/) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**[docs](https://machinefi.github.io/trio-retina/)** · **[quickstart](#-quickstart)** · **[the world-model stack](#-the-world-model-stack)** · **[benchmark](BENCHMARK.md)** · **[examples](examples/)** · **[notebooks](notebooks/)**

</div>

*A lightweight, model-agnostic **computer-vision pipeline** for **object detection & tracking** that emits structured **events** — zone intrusion, line-crossing, dwell, people-counting — from **YOLO**, **VLM**, or **Grounding DINO** detectors over video, files, or **RTSP**. Runs on CPU at the **edge**; feeds **digital twins**, dynamics models, and LLMs.*

> Just want camera events (zone intrusion, line-crossing) pushed to a webhook? → jump to the [5-line quickstart](#-quickstart), or copy [`examples/rtsp_to_webhook.py`](examples/rtsp_to_webhook.py).

## 👋 hello

**Trio Retina** (Retina for short) turns raw signals — video, sensor — into a **queryable world-state**: readable **events** (`zone.enter`, `dwell`, `line.cross`) *plus* a standardized **latent** `vec` channel on the same records, on one small model-agnostic standard. The latent channel is a real, serializable interface (attach your own embedding — see [`examples/latent_vec.py`](examples/latent_vec.py)), and the automatic *producers* now ship: `DinoV2Embedder` fills per-object `entity.vec` and `VJepa2Embedder` fills the scene latent `ws.scene`. Bring any model (YOLO, V-JEPA, DINO, a VLM, or none); Retina assembles its output into state a dynamics model, rule engine, or LLM can consume — and a small example dynamics model [imagines the future off that state](#-the-world-model-stack).

Think **OpenTelemetry for perception** — it doesn't build the sensors, it normalizes any of them into one state. In world-model terms it's the **encoder** (`s = Enc(x)`), and *only* the encoder; dynamics and policy build on top. → see [`DESIGN.md`](DESIGN.md).

## 💻 install

```bash
pip install trio-retina            # core: numpy only
pip install 'trio-retina[yolo]'    # + Ultralytics YOLO adapter
pip install 'trio-retina[video]'   # + OpenCV frame source (files / RTSP / webcam)
pip install 'trio-retina[all]'     # everything
```

## 🔥 quickstart

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

**▶ with a real model + video** — `pip install 'trio-retina[yolo]'` (add `[video]` for the frame source), then point it at your clip:

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
    # {"type":"zone.dwell","t":1718254799.8,"src":"cam_01","id":42,
    #  "label":"person","zone":"dock","dur":31.0,"conf":0.91}
```

More no-model examples ship with the source (not the wheel) — `git clone` the repo and run `python examples/quickstart.py` (the forecast / video demos need `[video]` + a clip).

**▶️ Or run it in your browser — no install:**

| notebook | what it shows |
|---|---|
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machinefi/trio-retina/blob/main/notebooks/01_quickstart_events.ipynb) | **quickstart** — detector → `zone` / `line` / `count` / `dwell` events + `validate()` |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machinefi/trio-retina/blob/main/notebooks/02_camera_to_webhook.ipynb) | **camera → webhook** — a restricted-zone alert pushed to your endpoint |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machinefi/trio-retina/blob/main/notebooks/03_from_supervision.ipynb) | **from Supervision** — pipe your existing `sv.Detections` straight in |

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

Retina imports no model — **any** detector plugs in, and out comes one standard event stream. That seam *is* the point:

| plug in any detector… | → | …out comes one `retina.event` stream |
|---|:---:|---|
| **YOLO** (Ultralytics: v5–v12, RT-DETR) | → | `{"type":"zone.enter", "id":42, "label":"person", …}` |
| **any VLM** (GPT-4o · Qwen-VL · Gemini · Claude) | → | `{"type":"line.cross", "dir":"a_to_b", …}` |
| **Grounding DINO** (open-vocab, no training) | → | `{"type":"zone.dwell", "dur":31.0, …}` |
| your existing **`sv.Detections`** (Supervision) | → | `{"type":"count.threshold", "n":12, …}` |
| any **`callable(image) -> [Detection]`** | → | …+ an optional latent `vec` on the same record |

Supervision gives you boxes on a screen; Retina turns *any* of those into a serializable state + event stream the next layer (dynamics, twin, agent) can consume. Batteries-included adapters:

- **YOLO family** — `YoloDetector("<weights>.pt")` (Ultralytics): YOLOv5/8/9/10/11/12, RT-DETR. Open-vocab via YOLO-World.
- **Open-vocab from text** — `GroundingDinoDetector(["forklift", "hard hat"])`, no training.
- **Any VLM** — `VlmDetector(client, prompt)` (Qwen-VL / Gemini / GPT-4o / Claude / local), as a detector or an event-source enricher.
- **Supervision interop** — `Detection.from_supervision(sv_detections)` ingests a Roboflow `sv.Detections`, so anything that already converts to Supervision pipes straight into Retina's event layer.
- **Latent producers (shipped)** — `DinoV2Embedder()` fills per-object `entity.vec` (frozen DINOv2, `pip install 'trio-retina[dino]'`); `VJepa2Embedder()` fills the scene latent `ws.scene` from a rolling clip (frozen V-JEPA 2 video encoder, `pip install 'trio-retina[vjepa]'`). Swap either underneath the same fixed state schema — see [`examples/world_model/multi_encoder.py`](examples/world_model/multi_encoder.py).

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

## 🌍 the world-model stack

Retina is the **encoder** (`s = Enc(x)`) in a world model. With the latent
producers shipped, the whole front-to-back seam is now demonstrable end to end —
on a synthetic scene, as a small but honest proof of concept ([`examples/world_model/`](examples/world_model/)):

**1 · swap the encoder, the state is constant.** The same pipeline, run three
ways — symbolic-only, `+ DinoV2Embedder` (per-object `entity.vec`), and
`+ VJepa2Embedder` (scene-level `ws.scene`) — yields the *identical* WorldState
schema; only which model filled the latent changes. → [`multi_encoder.py`](examples/world_model/multi_encoder.py)

**2 · a dynamics model imagines the future off that state.** A small transformer
trained offline on recorded `WorldState` sequences predicts where each entity is
headed, and rolls out *imagination* trajectories inside the learned model. The
honest ablation — does Retina's appearance latent actually help? — on **held-out**
data with **real DINOv2** vecs (Mac Studio, MPS), mean held-out 7-step position
error (px, lower is better):

| dynamics input | 7-step error |
|---|---|
| constant-velocity baseline | 7.68 px |
| learned, pos-only | 1.45 px |
| **learned, pos + appearance latent** | **1.33 px** |

**The latent channel measurably improves prediction: +83% over constant-velocity,
+8% over pos-only at horizon 7** — and the edge *widens with the horizon*, because
that's where local velocity runs out and object *type* (legible only from
appearance) decides the future. → [`dynamics.py`](examples/world_model/dynamics.py), full grid in [`BENCHMARK.md`](BENCHMARK.md)

<p align="center"><img src="https://raw.githubusercontent.com/machinefi/trio-retina/main/media/world_model_soccer.gif" width="840" alt="Left: raw broadcast soccer clip. Middle: a WorldState arrow. Right: a top-down tactical radar where each player is a team-coloured dot with a brand-indigo predicted next run and a faint gray past trail."></p>

> **Raw video → one standardized Retina `WorldState` → predicted player runs.** Left is a real broadcast clip (Roboflow's MIT-licensed [`sports`](https://github.com/roboflow/sports) sample, originally DFL Bundesliga). It goes through a real YOLO detector + tracker and a frozen DINOv2-small appearance encoder, and comes out as one model-agnostic `WorldState`; the right panel renders that state as a **top-down tactical radar**, and the small dynamics transformer — trained offline on those sequences — draws each player's **predicted next run** ahead in brand indigo (faint gray = where they came from). Teams are coloured by clustering the players' DINOv2 appearance vectors into two groups — the latent knows who's who. The radar is a stylized perspective top-down (no Roboflow pitch-keypoint weights on this host, so a fixed homography from the clip's pitch landmarks, not per-frame). Honest by design: player motion is stochastic, so at this short horizon the learned model roughly *ties* a constant-velocity baseline on held-out error — the appearance latent's *measurable* win lives in the cleaner synthetic ablation above, not on free-running humans. Real pipeline, end to end — [`examples/world_model/soccer/`](examples/world_model/soccer/). The synthetic car rollout (held-out, where the latent earns its keep) lives in [`make_demo_gif.py`](examples/world_model/make_demo_gif.py) · [`media/rollout.png`](examples/world_model/media/rollout.png).

**3 · front + back compose through one standard.** Any encoder in front, any
dynamics behind, meeting on one serializable state — a pip-installable world-model
seam you can run in one script. → [`end_to_end.py`](examples/world_model/end_to_end.py)

```bash
pip install 'trio-retina[dynamics,dino]'
python examples/world_model/dataset.py --n 12 --len 24 --out examples/world_model/data/sequences.json
python examples/world_model/end_to_end.py   # encoder → WorldState → dynamics → imagined rollout
```

> Honest scope: a synthetic scene, a tiny model, reproducible on MPS with
> run-to-run variance. The *producers* ship; the *trained dynamics* is a small
> example, not a product. The point is the seam — that front and back compose
> through one standardized state.

## 🎬 demos

Two more ways Retina's state feeds the next layer — same standard, different consumers:

<table>
<tr>
<td width="50%" valign="top">

**Forecast — the dynamics layer on Retina**

<img src="https://raw.githubusercontent.com/machinefi/trio-retina/main/media/retina_demo.gif" width="100%" alt="Trio Retina: YOLO object tracking with two dynamics models forecasting entity trajectories from one world-state — gray constant-velocity baseline vs magenta learned model">

One world-state → **two dynamics models forecast** where each entity is headed off the *same* state (gray = constant-velocity, magenta = learned). A dynamics model eats structured **state**, not pixels. → [`examples/forecast/`](examples/forecast/)

</td>
<td width="50%" valign="top">

**iTwin.js — a live, predictive digital twin**

<img src="https://raw.githubusercontent.com/machinefi/trio-retina/main/examples/itwin/media/retina_itwin_demo.gif" width="100%" alt="Trio Retina perception events and forecast arrows rendered live on a Bentley iTwin.js digital twin (Baytown plant)">

Retina's entities, forecast arrows, and `retina.event` alerts on a real Bentley **iTwin.js** iModel (Baytown), one neutral JSON contract, fully headless — it gives the twin *live eyes*. → [`examples/itwin/`](examples/itwin/)

</td>
</tr>
</table>

<details>
<summary>All examples</summary>

The examples live in this repo (not in the installed wheel) — `git clone` to run them. The top-level quickstarts run with **no model and no GPU** (synthetic detections):

```bash
python examples/quickstart.py          # zone / line / count / dwell events
python examples/three_apps.py          # one stream -> security, retail, safety
python examples/any_model.py           # swap the detector, rest unchanged
python examples/gate_savings.py        # a cheap gate cuts detector calls 100 -> 23
python examples/pipeline_compose.py    # compose with | (n8n without a GUI)
python examples/rtsp_to_webhook.py     # camera -> restricted-zone alert -> webhook
python examples/from_supervision.py    # ingest a Roboflow sv.Detections pipeline
python examples/latent_vec.py          # populate the latent vec channel by hand
python examples/dino_embeddings.py     # REAL DINOv2 per-object vecs (needs [dino])
```

Real-footage / dynamics demos need a clip and the extras — `pip install 'trio-retina[all]'`:

```bash
python examples/yolo_video.py v.mp4    # YOLO on a video file
examples/forecast/                     # dynamics layer on the WorldState stream (needs [video] + a clip)
examples/itwin/                        # events + forecasts on a Bentley iTwin.js iModel
```

The **world-model stack** lives in [`examples/world_model/`](examples/world_model/) (needs `[dynamics]`, plus `[dino]`/`[vjepa]` for real encoders):

```bash
python examples/world_model/multi_encoder.py   # swap encoder, state schema stays constant
python examples/world_model/dynamics.py        # train + the honest appearance ablation
python examples/world_model/benchmark.py       # the front/back-end benchmark grid → BENCHMARK.md
python examples/world_model/end_to_end.py      # encoder → WorldState → dynamics → imagined rollout
```
</details>

**Send events anywhere.** `WebhookSink(url)` POSTs each event as JSON (stdlib urllib, no `requests`); `JsonlSink(path)` streams to a file. For a live camera, `video_frames(src, live=True)` reads RTSP / HLS / webcam with wall-clock timestamps — see [`examples/rtsp_to_webhook.py`](examples/rtsp_to_webhook.py).

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

**Dual state.** The same entities on two linked channels: *symbolic* (readable `events` / entity records, for rules / LLMs / dashboards) and *latent* (optional model-tagged embeddings, for a downstream dynamics model). Symbols you can read; vectors a model can predict on. The latent channel is a standardized, serializable interface — populate `entity.vec` with your own embedding ([`examples/latent_vec.py`](examples/latent_vec.py)), or let a built-in producer fill it: `DinoV2Embedder` (per-object) and `VJepa2Embedder` (scene-level) both ship today.

**vs DeepStream / Holoscan** — same good ideas (event semantics, metadata model, composable graph), none of the weight:

| | DeepStream / Holoscan | **Retina** |
|---|---|---|
| Install | CUDA + TensorRT + containers | `pip install trio-retina` |
| Hardware | NVIDIA / Jetson locked | any machine — CPU is fine |
| Model | tied to the NV stack | **bring any model** (or none) |
| Shape | a platform you build *inside* | a library you `import` |
| Core deps | a lot | **numpy** |

**vs Supervision** — Supervision turns a model's output into detections + overlays (great toolbox, ends at the screen). Retina is a level up: it emits a serializable **state + event stream** that the *next* layer (dynamics, twin, agent) consumes. We compose Supervision / detectors, not compete with them.

Full rationale, references, and the world-model stack: [`DESIGN.md`](DESIGN.md).
</details>

## 🗺️ roadmap

Early but real (`v0.2.1`). Stable: the event layer + JSON Schema/validator, the composable pipeline (`|` / list / JSON), YOLO + open-vocab + VLM detectors (plus `from_supervision` interop), IoU + Norfair trackers, and jitter-robust rules (`exit_grace_s` · `anchor` · `min_frames`).

Next: ByteTrack / OC-SORT · `proximity` / `anomaly` events · VLM-as-event-source · Kafka / MQTT sinks · **more encoders** behind the latent channel · **model-based RL / latent-rollout imagination** on the learned state · growing the [front/back-end benchmark](BENCHMARK.md). See [`CHANGELOG.md`](CHANGELOG.md).

Retina is the open **perception encoder** extracted from [Trio](https://machinefi.com); the layers above (dynamics, policy / judgment) are Trio's commercial platform. Retina is, and stays, model-agnostic and free.

## 🤝 contributing

Contributions that keep Retina small and beautiful are very welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md) for dev setup and how to add a detector / tracker / rule / sink. By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md); to report a vulnerability see [`SECURITY.md`](SECURITY.md).

## license

[Apache-2.0](LICENSE).
