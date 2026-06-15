# Retina — design notes

**What Retina is:** the **perception encoder** of a world model. In the standard
decomposition (encoder → dynamics → decoder; Ha & Schmidhuber, Dreamer, V-JEPA 2,
LeCun, Zheng & Niyato 2026), Retina is the **encoder** — `s = Enc(x)`: raw
real-world signals → state. Models below (YOLO / VLM / V-JEPA), applications and a
future dynamics model above. Honest scope: it produces **state**, not the dynamics
or the policy; we say "encoder," not "world model."

**Dual output (state has two linked channels on the same entities):**
- **symbolic** — readable, queryable events / entity records. The model-agnostic
  standard ([`SPEC.md`](SPEC.md)); feeds rules / LLM-judges as JSON.
- **latent** — optional `vec` on the same records: small per-object ReID vector
  (inline) + large scene-level V-JEPA vector (by-reference), model-tagged. Feeds a
  downstream dynamics model. (The shape object-centric / neuro-symbolic world
  models converged on — Slot Attention, Concept Embedding Models, STTran.)

**North star:** small & beautiful, developer-friendly, pluggable. Pure-Python
core (numpy only). No hardware lock, no model lock, no install hell, not a
platform you build *inside* — a library you `import`. Claim only the rung we're
on: encoder now; the **dynamics model** (state + action → next state) is the next
product, built *on* Retina, and the point at which "world model" is earned.

## What we absorbed (and what we dropped)

From **NVIDIA DeepStream** (the industrial gold standard for video→events):

- **ABSORB — append-only metadata tree.** `Frame → Detection`, each stage
  *enriches and never overwrites*. Keep detector bbox and tracker bbox
  separately. An open `user: dict` slot on every node = extension without
  forking the schema (their `NvDsUserMeta`, for free under Python GC).
- **ABSORB — the analytics/rules engine (`nvdsanalytics`).** zone / line
  (directional) / count / dwell, authored at **normalized coords** (resolution
  independent), the **rule's id is the output key**, results at both per-object
  and per-frame granularity.
- **ABSORB — the event ontology, minimized.** `sensor / object / event{type}`
  with a small closed verb set (`entry/exit/dwell/...`). We flattened it to a
  JWT-style format (see `SPEC.md`).
- **SKIP** — GStreamer, NVMM/GPU buffers, metadata pools, the `pyds` binding
  layer, TensorRT. 80% of the weight, 0% of the concept.

From **NVIDIA Holoscan**:

- **ABSORB — pluggable stages behind tiny protocols** (`Detector`, `Tracker`,
  `EventRule`, `EventSink`), wired explicitly. Topology separate from behavior.
- **ABSORB (roadmap) — condition-based triggering.** Every trigger (data
  available, periodic, count, gate) as one uniform `Condition`; an event rule is
  "a condition over a semantic stream." Deferred from 0.1 to stay minimal; it is
  the runtime spine for 0.2 (and the natural home for VLM-call gating).
- **SKIP** — GXF, CUDA-resident tensors, UCX/RDMA, C++ codelets, compiled
  bindings, thread-pinning.

From **academia**:

- **ABSORB now** — promptable/open-vocab detectors as adapters (YOLO-World,
  Grounding-DINO, T-Rex2) → events for any object described in words, zero
  training. Pure-Python trackers (Norfair) and ByteTrack/OC-SORT behind the
  `Tracker` protocol (MASA's "any detector → tracker" pattern). Prediction-error
  / embedding-novelty (V-JEPA, Liu 2018) as a cheap `anomaly` primitive and a
  detection **gate** hook.
- **ABSORB into the schema roadmap** — MEVA/ActEV activity vocabulary, VidVRD
  predicate structure (proximity/interaction), Vid2Seq "time-tokens →
  constrained JSON decode" so a VLM emits `retina.event` objects directly.
- **NOT here — the distillation/arbitrage engine** (Autodistill, Soft-Teacher,
  ShareGPT4V amplification, SAM data engine; FrugalGPT/VideoAgent/Cerberus
  gating). That is the commercial / Spark layer. Retina core stays the pure
  event layer; it only exposes the `detector` and `gate` seams those plug into.

## Architecture (0.1)

```
frames ─(Detector)→ detections ─(Tracker)→ tracks ─(EventRule[])→ events ─(EventSink[])→ out
            ▲             ▲                     ▲                      ▲
       any model     IoU / Nortrack /      zone / line /          jsonl / webhook /
       (or VLM)      ByteTrack (plug)      count / dwell           kafka (plug)
```

Every arrow is a tiny Protocol. The data unit is the append-only `Frame`
(`retina/events.py`); the wire unit is the `Event` (`SPEC.md`).

## Layer boundary (hard rules)

- **App-agnostic:** Retina emits only generic primitives. No `shoplifting`,
  no `PPE_violation`, no `goal` — domains compose those *above*, from primitives.
- **Model-agnostic:** Retina imports no model. A detector is any callable
  `frame -> list[Detection]`. VLMs/world-models plug in the same seam.

## Status / roadmap

- **0.1 (now):** event format + geometric primitives (zone/line/count/dwell),
  IoU tracker, JSONL/webhook sinks, YOLO + callable detector adapters.
- **0.2:** Condition-based runtime + gate hook; Frame as a first-class stream;
  Norfair/ByteTrack adapters; `proximity`/`anomaly` primitives.
- **later:** VLM-as-producer (constrained JSON decode), schema vocabulary
  growth (MEVA/VidVRD), Kafka/MQTT sinks.
