# CSI world model through Retina — a gap-finding demo

A small **CSI / RF world model** built *through* Retina's abstractions, to find out
exactly what the framework is missing for field-like (non-vision) sensing. It
reproduces the *essence* (small scale) of two papers:

- **arXiv:2409.10045** *Learning Latent Wireless Dynamics from CSI* — a JEPA that
  encodes CSI → latent `z`, predicts the **next latent** `z_{t+1}` (no raw
  reconstruction); the latent self-organizes into a metric **channel chart**.
- **arXiv:2603.20048** *Structured Latent Dynamics via Homomorphic World Models* —
  an **action-conditioned** transition where the action `a_t` is the **user's
  velocity**, parameterized as a homomorphic (Lie-algebra) update on the latent.

Everything is **synthetic** (a documented CSI forward model), fully **offline**,
numpy-only except the torch dynamics, and it **actually runs end-to-end**.

## Files

| file | what it is |
|---|---|
| `csi_data.py` | Synthetic CSI forward model: a moving point-scatterer (the subject) + weak static multipath in a 6×5 m room → complex CSI `H_t` (32 subcarriers × 8 antennas); the subject walks with a velocity = the **action**. numpy-only. |
| `csi_state.py` | Assembles a **real** `retina.WorldState` per CSI timestep — global channel latent → `ws.scene`, the subject → an `Entity` whose **metric position rides the native `Entity.locus`** (a typed world-frame coordinate, distinct from the pixel `bbox`). |
| `csi_dynamics.py` | Thin CSI adapter over the **shared** [`../latent_dynamics.py`](../latent_dynamics.py) — picks the CSI defaults (velocity action, channel-chart latent) and re-exports its builder/losses/rollout. |
| `../latent_dynamics.py` | The reusable **action-conditioned JEPA** (torch), signal-agnostic (`feat_dim`/`latent_dim`/`action_dim`): encoder → latent, homomorphic (skew-symmetric/orthogonal) action transition `z_{t+1}=exp(A(a))z_t + b(a)`, JEPA latent loss + VICReg anti-collapse, latent-space `rollout`. Sits beside the shipped `dynamics_model.py` (the appearance-ablation example). |
| `train_csi.py` | End-to-end: data → Retina state → train → **fusion ablation** → **channel-chart probe** → **imagination rollout vs constant-velocity baseline**. Prints real numbers. |
| `test_csi_demo.py` | Smoke tests (forward model, real-types state round-trip, one JEPA train step, norm-stability of the homomorphic update). |

## Run it

```bash
pip install -e '.[dynamics]'          # torch + matplotlib (matplotlib unused here)
cd examples/world_model/csi
python train_csi.py                   # ~25 s CPU
pytest test_csi_demo.py               # 6 tests
```

## Real run output (seed 0, CPU, 60 epochs)

```
(1) MULTIMODAL FUSION ABLATION — 5-step next-latent L2 error (lower=better):
    action-conditioned (CSI + velocity) : 1.0097
    CSI-only           (action ignored) : 3.7279
    => fusing the velocity action improves next-latent by +72.9%   VERDICT: HELPS

(2) CHANNEL CHART — linear probe latent->position fit error: 0.325 m   (room is 6×5 m)

(3) IMAGINATION ROLLOUT — 14 steps, latent rolled under actions, position via probe:
    step   world-model(m)   const-vel(m)
       0          0.330           0.000
       7          0.637           0.662     <- they cross here
      14          0.899           2.078
    mean          0.619           0.794     VERDICT: world-model beats const-vel by 22.0%
```

Robustness (seeds 0/1/2): fusion helps **+69–73%** every seed; chart **0.30–0.33 m**
every seed; at the 14-step horizon the world-model beats constant-velocity on
**all three** seeds. The world-model's edge **grows with the horizon** — the
slow-walk constant-velocity baseline is strong for ~7 steps, then diverges while the
action-conditioned latent rollout stays flat. That horizon-dependent win is exactly
the claim of arXiv:2409.10045.

> **Forward-model honesty.** With a *cluster of bright static reflectors* the
> CSI→room map became many-to-one and position stopped being decodable even by a
> nonlinear probe (~1.85 m = the mean-prediction baseline). We keep the subject's
> LOS path dominant and the reflectors weak — a LOS-dominant WiFi-sensing geometry
> — so the chart is learnable. This is a real property of the physics, documented
> in `csi_data.py`, not a fudge.

---

# GAP ANALYSIS — what Retina lacks for a CSI / field world model

The point of the demo. Each gap is tagged **(a)** what's missing, **(b)** layer
(CORE = `worldstate.py` / DYNAMICS-layer / example-only), **(c)** invasiveness,
**(d)** recommendation.

### GAP 1 — Field / non-entity metric position ✅ CLOSED (`Entity.locus`)
- **(a)** CSI is a **field** measurement of a whole room, not "objects with
  bounding boxes." The moving subject has **no pixel bbox**; its real state is a
  position **in metres**. Previously that had no typed home and was stuffed into the
  untyped `attrs` dict.
- **(b)** CORE (`worldstate.py`).
- **(c)** Additive / backward-compatible — `bbox` was already `Optional`.
- **(d)** **DONE.** `Entity.locus: tuple[float,...] | None` now carries a **metric
  position in a world/scene coordinate frame** (units/frame defined by the
  producer, e.g. metres) — distinct from the pixel-space `bbox`. `csi_state.py`
  puts the subject's room-frame position straight on `locus`, no `attrs` abuse. It
  serializes omit-empty (vision payloads with no `locus` are byte-for-byte
  unchanged). The remaining "field/source entity kind" (a `type` vocabulary for
  non-vision sources) is left to the producer's `type` string — no schema change
  needed. Whole-field latents already had their clean home in `ws.scene`.

### GAP 2 — Action-conditioned dynamics interface ✅ CLOSED (dynamics) · CORE deliberately NOT touched
- **(a)** A world model is `(state, action) → next state`. The original shipped
  dynamics `build_model(...)` / `rollout(...)` take **no action argument** — they
  model `p(s_{t+1}|s_t)` only. The homomorphic-world-model paper conditions on the
  velocity action.
- **(b)** DYNAMICS-layer (the predictor signature). We **judged the CORE side out
  of scope on purpose**: an action is a *transition input*, not state, so it does
  **not** belong on the `WorldState` snapshot — adding a `WorldState.action` field
  would conflate state with transition. The action is consumed out-of-band by the
  dynamics, exactly like a control input.
- **(c)** Low — the dynamics interface is example-level.
- **(d)** **DONE (dynamics).** `latent_dynamics.build_jepa(...)` and its
  predictor/loss/rollout all take `a_t`; the signature is `predict(z_t, a_t)`. The
  CSI demo still records the applied velocity in `attrs["vel_action_m"]` purely so
  the serialized frame self-describes — but it is *not* a state field.

### GAP 3 — No JEPA latent-prediction dynamics head ✅ CLOSED (`latent_dynamics.py`)
- **(a)** The original shipped dynamics predicts a **position delta `(dx,dy)`** with
  an MSE-on-positions loss. A JEPA predicts the **next latent** `z_{t+1}` with a
  latent-space loss and needs anti-collapse (target/EMA encoder + VICReg).
- **(b)** DYNAMICS-layer.
- **(c)** A different head + loss + EMA target tower, but **zero CORE change** — it
  consumes the same `vec` channel.
- **(d)** **DONE.** The action-conditioned JEPA now ships as a reusable, signal-
  agnostic example variant: [`../latent_dynamics.py`](../latent_dynamics.py)
  (`feat_dim`/`latent_dim`/`action_dim`), sitting beside the appearance-ablation
  `dynamics_model.py`. The CSI demo just imports it. It carries an EMA target
  encoder, a homomorphic action transition, the latent-space loss, and VICReg.

### GAP 4 — No CSI / RF source adapter 🟠
- **(a)** `retina/sources.py` is **video-only** (OpenCV `VideoCapture`, frames as
  `HxWx3`). There is no adapter for a stream of CSI tensors / IQ samples / any non-
  image sensor. We bypassed sources entirely and fed numpy arrays directly.
- **(b)** Example-level to add one adapter; a small CORE generalization to make
  "source" formally sensor-agnostic.
- **(c)** Low. Sources are already "just an iterable of `(frame, t)`," so the
  contract is close — but everything downstream assumes `frame.image` is an image.
- **(d)** **Recommended (low priority).** Ship a `csi_frames(...)` example adapter;
  optionally generalize the `Frame` so `frame.signal` can be a non-image tensor
  without pretending to be a picture.

### GAP 5 — No multimodal Vec-fusion API 🟠
- **(a)** Retina has **two latent slots** (`entity.vec` per-object + `ws.scene`),
  but **no operation to fuse channels** into a joint state the dynamics consumes.
  The existing `with_appearance` ablation hard-concatenates motion + one `vec` *in
  the dynamics model itself*. We did the same by hand (CSI latent + velocity
  action). There's no `fuse([vec_a, vec_b]) -> vec` or typed multi-vec carrier.
- **(b)** Could be a small CORE helper (a `Vec.concat` / fusion node) or a DYNAMICS-
  layer convention.
- **(c)** Low.
- **(d)** **Recommended.** A tiny tagged-fusion helper (concatenate + re-tag
  `{model, dim}`) would standardize multimodal state assembly; the model-tagging
  discipline already in `Vec` makes this clean.

### GAP 6 — Slot-count rigidity (`N_SLOTS` hardcoded) 🟡
- **(a)** `dynamics_model.py` hardcodes `N_SLOTS = 2` and assigns entities to fixed
  slots. CSI here is **one field latent per timestep** (no slots at all); other RF
  scenes have a *variable* number of sources. The fixed-slot grid doesn't fit.
- **(b)** DYNAMICS-layer (it's an example today).
- **(c)** Low.
- **(d)** **Recommended.** Make slot count a config / support a slot-free
  (single-field) mode. Our `csi_dynamics.py` is already slot-free, which is the
  natural shape for a field latent.

### GAP 7 — Large latent dims & `ref` carriers, lightly exercised 🟢
- **(a)** `Vec` supports `ref` (by-reference, for large latents) and arbitrary
  `dim` — good. But the *dynamics* assumes a small inline `vec`; a by-`ref` CSI/
  V-JEPA latent (1024+ dims) has no resolution path in the dynamics loader. Not a
  schema gap, an integration gap.
- **(b)** DYNAMICS-layer / example.
- **(c)** Low.
- **(d)** **Optional.** Document/resolve `ref` latents in the dynamics dataset prep.

---

## What worked cleanly (validates Retina's design)

These accepted CSI with **zero friction** — evidence the wide/dual-channel design
generalizes past vision:

- **`ws.scene` is a perfect home for a whole-field latent.** CSI is a room-level
  field measurement; `scene` is "a scene latent with no box." This was the single
  best fit in the exercise — the global channel latent dropped straight in.
- **The dual symbolic + latent split held.** We kept a readable symbolic core
  (position, velocity in metres) *and* a model-tagged `vec`, never collapsed — the
  same discipline as the vision path.
- **`Vec` model-tagging is genuinely model-agnostic.** A `"csi-jepa/v0"` latent
  rides the exact same `{model, dim, dtype, ref, values}` carrier as a
  `"dinov2-small"` one, no schema change. The "import no model" thesis paid off:
  an RF latent is just another tagged vector.
- **`to_dict()` / omit-empty serialization round-trips RF state** unchanged — the
  CSI `WorldState` serializes to 557 bytes and parses back losslessly (tested).
- **The `vec` channel as the dynamics interface is the right seam.** Swapping a
  DINOv2 appearance vec for a CSI JEPA latent required **no change to the state
  schema** — only a new encoder and a new dynamics head. The encoder really is the
  swappable interface the design claims.

**Bottom line.** Retina's **latent channel and dual-state schema generalized to RF
with no core change** — a strong validation. The two highest-value gaps this demo
surfaced have now been **closed**: (1) field/metric position is a first-class typed
`Entity.locus` (one additive, backward-compatible CORE field), and (2) the
action-conditioned JEPA latent dynamics ships as a reusable example variant
([`../latent_dynamics.py`](../latent_dynamics.py)) beside the appearance-ablation
model. Together they turn Retina from a vision perception encoder into a substrate
for physical-field world models — the same `WorldState`/`Vec` that carried DINOv2
appearance now carries a WiFi CSI channel latent, with the position of a field
source typed natively.
