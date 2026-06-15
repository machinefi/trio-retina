# retina.event/0.1 — the event interchange format

A standard, model-agnostic, app-agnostic format for *what happened* in a stream —
the **symbolic** half of the state a perception encoder produces. Any model below
(YOLO, a VLM, a V-JEPA world model) writes it; any application *or downstream model*
above reads it. An optional **latent** channel (see [Dual state](#the-latent-channel--dual-state))
rides alongside for learned models.

Designed to be **boring and tiny**, on purpose — like a JWT. The smallest valid
event is three fields:

```json
{"type": "line.cross", "t": 1718254799.8, "src": "cam_01"}
```

Add fields only when you have them. Nothing is nested. Nothing is required
except the three below.

## Required fields

| field | type | meaning |
|---|---|---|
| `type` | string | event type — one of the primitives below (`domain.verb`) |
| `t` | number \| string | when it happened. Epoch seconds (preferred) or RFC3339. For a span, this is the **start**. |
| `src` | string | source / sensor id (which camera, stream, sensor) |

## Registered optional fields

All optional. **Omit any field you don't have** — that's what keeps events tiny.

| field | type | meaning |
|---|---|---|
| `id` | int | track id of the subject |
| `label` | string | object class (`person`, `car`, …) |
| `zone` | string | zone or line id the event refers to |
| `dur` | number | duration in seconds (for spans like dwell). Absent ⇒ instantaneous. |
| `dir` | string | direction (`in`, `out`, `a_to_b`, …) |
| `n` | int | a count (for `count.*` events) |
| `conf` | number | confidence, 0..1 |
| `box` | [x1,y1,x2,y2] | bounding box of the subject, pixels |
| `by` | string | the model/pipeline that produced it (`yolo11n+iou`, `gpt-vlm`) |
| `frame` | int | source frame index (evidence) |
| `clip` | string | URI to an evidence clip/image |
| `eid` | string | globally-unique event id (for dedup/idempotency) |
| `vec` | object | optional latent attached to the subject — see [Dual state](#the-latent-channel--dual-state) |

## Custom fields

Need something not above? **Just add a key.** Namespace it to avoid collisions
(`acme.shift`, `x_temperature`). The same way you add a private claim to a JWT.

```json
{"type":"zone.dwell","t":1718254799.8,"src":"cam_01",
 "id":42,"label":"person","zone":"north_dock","dur":31.0,"conf":0.91,
 "acme.shift":"night"}
```

## The latent channel — dual state

Retina is a perception encoder, and its state has **two linked channels on the same
entities**: the **symbolic** core above (readable, queryable, the standard) and an
optional **latent** channel for downstream learned models (a dynamics model, an RL
policy). Symbols you can read; vectors a model can predict on — never collapsed.

A latent rides as a `vec` object, **always model-tagged**:

```json
"vec": {"model": "osnet-reid",  "dim": 512,  "dtype": "fp32", "values": [ ... ]}      // inline (small)
"vec": {"model": "v-jepa2-vitl","dim": 1024, "dtype": "fp16", "ref": "vec://abc123"}  // by-reference (large)
```

- **inline** small single-model vectors (ReID 128–512); **by-reference** (`ref`) large
  or re-embeddable ones (a V-JEPA scene vector, ~1024–1408 dims).
- **always tag** `{model, dim, dtype}` — a FaceNet-128 and a V-JEPA-1024 can't share an
  index; the tag says what produced it.
- the symbolic core stays the **model-agnostic standard**; the latent is **model-coupled**
  and optional.

The fuller **state** (roadmap) is a `WorldState`: a *set* of entities (each = a symbolic
record + optional `vec`), a scene-level `vec`, and typed relations — the shape
object-centric / neuro-symbolic world models converged on. Fusion: detector+tracker →
symbolic core + per-object ReID `vec`; frozen V-JEPA → scene `vec` (+ optional
ROI-pooled per-entity `vec`).

## Primitive event types (0.1)

A small, **closed** vocabulary. Generic, model-agnostic, app-agnostic. Domains
compose meaning *above* this layer (a `line.cross` becomes "intrusion" or
"customer entered" in the application, never here).

| type | fires when | typical fields |
|---|---|---|
| `zone.enter` | a track enters a zone | `id, label, zone` |
| `zone.exit` | a track leaves a zone | `id, label, zone, dur` |
| `zone.dwell` | a track has stayed in a zone ≥ threshold | `id, label, zone, dur` |
| `line.cross` | a track crosses a line | `id, label, zone, dir` |
| `count.threshold` | object count crosses a threshold | `n, zone` |

## Serialization

- A single event is a JSON object.
- A stream of events is **JSON Lines** (one event per line) — greppable,
  appendable, replayable.
- Producers SHOULD omit null/empty fields.

## Validation

A formal JSON Schema (draft 2020-12) ships as `retina/event.schema.json`. In
Python, validate without extra deps:

```python
from retina import validate, is_valid
validate(event)   # -> [] if valid, else a list of problems
```

## Versioning

The format is identified by `retina.event/<major>.<minor>`. New primitive types
and registered fields are added in minor versions; removals/renames bump major.
Consumers MUST ignore unknown fields (forward compatibility).

## Roadmap (not in 0.1)

Reserved for later minor versions — listed so the vocabulary grows on a plan,
not ad hoc. Grounded in existing taxonomies (MEVA/ActEV activities, VidVRD
subject–predicate–object relations) rather than invented:

- `appear` / `disappear`, `state.change`, `anomaly`
- `proximity` (two tracks within distance), `interaction` (predicate between two
  tracks: `holds`, `next_to`, `transfers`, …)
- per-frame aggregate snapshots (`count.snapshot`)
