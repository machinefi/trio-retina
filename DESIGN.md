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

## Where Retina sits in the world-model stack (and the L1 axis)

The Dreamer / world-model line (World Models 2018 → DreamerV3 → DreamerV4) stacks
into ~7 layers. Retina's position — and what it deliberately does *not* do:

| layer | Dreamer line | Retina |
|---|---|---|
| **L1 perception / tokenizer** | VAE → RSSM latent → causal tokenizer (opaque latent) | **a new kind of L1** (see two axes below) |
| **L2 dynamics** | RSSM → block-causal transformer + shortcut forcing | not us — compose a decoder-free engine (TD-MPC2 / V-JEPA 2-AC) |
| **L3 reward / task** | reward head + task tokens | mostly N/A (operations have no single reward); ≈ the domain layer above |
| **L4 imagination** | latent → pixel-level rollouts | roadmap: forecast / what-if *on the state* (the dynamics product) |
| **L5 policy / value** | ES → actor-critic → PMPO | mostly skipped — we forecast, we don't control |
| **L6 training pipeline** | pretrain WM → finetune → imagination RL (decoupled) | same decoupling: one state, reused across apps |
| **L7 scaling / efficiency** | 2B params, single-GPU real-time | inverted: small, edge, cheap — symbolic, not scaled |

So Retina **owns a new L1**, treats **L4 as the next product**, mostly skips **L5**,
shares **L6's decoupling**, and **inverts L7** (small/edge, not big/GPU).

### The L1 has two orthogonal evolution axes

The usual L1 story — *reconstruction → masked autoencoding, for higher information
density* — is only **one axis**. There are two, and they're orthogonal:

```
  DEEP ▲  better opaque latent, co-designed with one dynamics model
       │  VAE → β-VAE → discrete codes (VQ) → MAE / causal tokenizer → JEPA
       │  opaque · end-to-end with L2 · single consumer
       │       ← the whole world-model community is on this axis
  ─────┼───────────────────────────────────────────────────────────►  WIDE
       │  Supervision → events → standard world-state
       │  model-agnostic · structured · serializable standard · multi-consumer · dual
       │       ← Retina is on this axis — and it is empty
```

- **Dreamer's L1 goes DEEP:** a better, opaque latent for *its own* dynamics —
  end-to-end co-designed with L2, single-consumer.
- **Retina's L1 goes WIDE:** model-agnostic, structured, a serializable *standard*
  state, consumed by many (rules, LLM, dynamics, humans, audit), dual symbolic+latent.

We are not behind on the deep axis; we are defining the wide one. The deep-axis
race (DINOv3, V-JEPA 2, causal tokenizers) is "a better latent" — we don't run it.

### What we take from the deep axis (for our latent channel)

- **Reconstruction → prediction (JEPA).** Prefer V-JEPA-style *predictive* latents
  over reconstruction latents in the `vec` channel — cheaper, more semantic.
- **Temporal tokenization.** Our event-sourced stream already "tokenizes time":
  continuous time compressed into discrete events / transitions.
- **Information density.** A JWT-style event is dense but lossy; the dual channel
  hedges — symbols for density/readability, the latent for the residual.
- **The payoff of agnosticism:** because we import no model, the deep axis's wins
  (DINOv4, V-JEPA 3, …) drop into our latent channel *for free*. We don't win the
  latent race — we plug in whoever does.

### The deep-vs-wide tension (honest)

Dreamer **co-designs L1+L2 end-to-end** → better prediction. We **decouple**
(model-agnostic) → reuse + a standard. We pay a small prediction-accuracy tax for
large reuse/interpretability gains; the latent channel is the hedge (L2 can still
have a learnable representation when it wants one).

### Retina's L1 roadmap (on the wide axis)

1. Richer structured state: entities + relations + scene-graph (→ a `WorldState`).
2. Better latent channel: ride the deep-axis winners (frozen V-JEPA 2 scene latent
   + per-object ROI latents), upgradeable as they improve.
3. Make "time → events" tokenization solid (event-sourcing as the native tokenizer).
4. Stay a **standard** wire format — multi-consumer is the whole point of "wide".

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
