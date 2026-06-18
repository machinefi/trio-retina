# Concepts

The whole mental model in one read. Retina turns a stream of images into a
standardized **state + event** stream — not boxes drawn on a screen. Everything
below is one small, serializable, model-agnostic contract.

## The data, bottom to top

```
Frame ──► Detection ──► Track ──► Event          (the retina.event standard)
  │           │            │         │
  image    one object   the same   "something happened":
  + ts     in one       object     zone.enter / zone.dwell /
           frame        over time  line.cross / count.threshold
                          │
                          └────────► WorldState  (the assembled snapshot:
                                     entities + relations + scene latent)
```

- **`Frame`** — the append-only unit flowing through the pipeline. Each stage
  *enriches* it (the detector fills `detections`, the tracker fills `tracks`,
  the rules fill `events`) and never overwrites an upstream field. It carries the
  raw `image`, a timestamp `t`, and a `src` id.
- **`Detection`** — one object found in one frame: a `label`, a `bbox`, a
  `confidence`. This is the **model-agnostic seam**: a *detector* is anything
  callable that maps a frame to a `list[Detection]`. YOLO, a VLM, Grounding DINO,
  or your own function — Retina never imports a model.
- **`Track`** — the same object given a stable `id` across frames (so "person 42"
  is the same person from frame to frame). Tracking is what makes `line.cross`
  and `dwell` meaningful: they need object identity.
- **`Event`** — a *transition* in the closed `retina.event` vocabulary
  (`zone.enter`, `zone.exit`, `zone.dwell`, `line.cross`, `count.threshold`).
  Tiny and flat like a JWT: three required fields (`type`, `t`, `src`), everything
  else optional and omitted when absent. See the [event spec](spec.md).
- **`WorldState`** — the *state* the events transitioned into: the set of
  **entities** present at one instant, their **relations**, and an optional
  **scene** latent. Where an `Event` is the delta, the `WorldState` is the frame
  it applies to. Stream it with `Pipeline.run_states()` alongside `run()`.

## The dual channel: symbolic core + latent `vec`

Every entity and every event carries a **symbolic core** (readable, model-agnostic
— `id`, `label`, `zone`, …) and an *optional* model-tagged **latent** `vec` on the
*same* record. The two are never collapsed: symbols you can read in a dashboard or
feed an LLM; vectors a downstream dynamics model can predict on.

The `vec` channel is a real serializable interface (`Vec(model=..., dim=..., values=...)`),
attachable by hand ([cookbook recipe 4](cookbook.md#4-attach-a-latent-vec)) or
filled automatically by a shipped producer (`DinoV2Embedder` per-object,
`VJepa2Embedder` scene-level).

## The pipeline

The stages above are composed into a linear chain. Same chain, three altitudes —
pick yours:

```python
# 1. `|` composition (LCEL / n8n-style, no GUI)
pipe = YoloDetector("yolo11n.pt") | IoUTracker() | ZoneRule(dock) | JsonlSink("e.jsonl")

# 2. explicit node list
from retina import Pipeline, DetectorNode, TrackerNode, RuleNode
pipe = Pipeline([DetectorNode(yolo), TrackerNode(), RuleNode(ZoneRule(dock))])

# 3. declarative JSON ("n8n without a GUI" — shareable, no code)
pipe = Pipeline.from_json("workflow.json")
```

Each step is a **Node** (`Frame -> Frame`, or `None` to drop the frame). The
shipped detector / tracker / rule / sink objects auto-wrap into the right Node, so
you usually only reach for an explicit Node to wrap a raw function of your own.
For the common detector → tracker → rules case, `Retina(detector=..., rules=[...])`
is sugar over a `Pipeline`.

## Where Retina sits

In world-model terms Retina is the **encoder** — `s = Enc(x)` — and *only* the
encoder: raw signals in, one standardized `WorldState` out. Perception backbones
(YOLO, DINOv2, V-JEPA 2, SAM, VLMs) feed it; dynamics and policy build on top of
the state it emits. Swap the model in front or the dynamics behind — Retina is the
constant in the middle.

![The world-model stack: perception backbones feed Trio Retina — the encoder and standardized WorldState — which dynamics and control build on top of.](https://raw.githubusercontent.com/machinefi/trio-retina/main/media/stack.png)

That's the differentiator versus a toolbox like Supervision: Supervision turns a
model's output into detections + overlays (great, ends at the screen); Retina
emits a serializable **state + event stream** the *next* layer (dynamics, a
digital twin, an agent) consumes. We compose detectors, not compete with them.

→ Try it now in the [cookbook](cookbook.md), wire your own pieces in
[extend](extend.md), or drive it from the [CLI](cli.md). The full rationale lives
in the [design notes](design.md).
