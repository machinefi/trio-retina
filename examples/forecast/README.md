# Forecast — the dynamics layer (L4) on top of Retina

This is a **demo of the next layer up**, not part of the Retina core. It shows a
dynamics model predicting the near-future **state** of a scene from Retina's
`WorldState` stream — and, more importantly, *why Retina is necessary* for it.

## The point: the dynamics eats state, not pixels

A dynamics model (TD-MPC2, an object-centric predictor) consumes a **structured
state vector**, not raw video — and has no real-world perception front-end. So to
forecast on real operations video you need something to turn *any* backbone's
output into that state. That something is Retina. The seam is model-agnostic on
**both** sides: swap the detector (YOLO → V-JEPA → DINO) *and* swap the dynamics
(`LinearForecaster` → `TDMPC2Dynamics`) without touching the rest.

> A skeptic asks "why not feed V-JEPA2 straight into the dynamics?" — because
> V-JEPA2 is all-in-one (its own encoder + predictor); using it *as the dynamics*
> would make Retina look redundant. We instead pick a dynamics that **needs** a
> state (TD-MPC2), and let V-JEPA2 be just one optional backbone feeding Retina's
> latent channel. Retina's value is the **structured, standard, multi-consumer**
> state — read on.

## Runs now (no model, no GPU)

```bash
python examples/forecast/quick_forecast.py
```

`video → Retina → WorldState stream → forecaster → predict t+H → score vs the
actual future state`. A constant-velocity baseline beats a no-motion baseline —
the WorldState is genuinely *dynamics-ready*. `dynamics.py` defines the swappable
`DynamicsModel` seam (`observe` / `predict`), the `LinearForecaster` baseline, and
a `forecast_error` metric.

## The wow demo (Mac Studio) — proves necessity, not just "it predicts"

Necessity can't rest on "it forecasts" (any encoder could feed a dynamics). It
rests on what only Retina gives. The full demo, on real operations video:

1. **One Retina state → three consumers at once:** a TD-MPC2 **forecast**
   ("#42 enters the restricted zone in ~3s"), live **rules** (zone/dwell events),
   and an **LLM-judge** reading the same state ("what's about to happen?"). One
   opaque V-JEPA2 latent serves one consumer; one Retina state serves many.
2. **Swap the backbone live** (YOLO ↔ V-JEPA): state, dynamics, rules, LLM keep
   working — Retina is the stable interface.
3. **Readable, per-entity forecast** you can act on — not an unreadable latent.

Heavy parts (TD-MPC2 + torch MPS, V-JEPA2 features, real video) run on the
**Mac Studio**; `TDMPC2Dynamics` in `dynamics.py` is the stable adapter seam.

## Status

- ✅ `dynamics.py` — `DynamicsModel` seam + `LinearForecaster` + `forecast_error`.
- ✅ `quick_forecast.py` — synthetic proof of the loop (no model/GPU).
- ✅ `forecast_video.py` — real-video baseline (YOLO). On a fixed-cam intersection:
  no-motion **63.1px → constant-velocity 39.9px** (−37%); the ~40px residual
  (turns, accel, interactions) is the TD-MPC2 target.
- ✅ `multi_consumer.py` — one Retina pass → three consumers at once (301 rule
  events + per-entity forecast + NL judge). Backbone swap is shown by `any_model.py`.
- ✅ `export_trajectories.py` + `train_dynamics.py` — export the WorldState tracks,
  then train a **learned dynamics** (small MLP, torch + MPS) on the **Mac Studio**
  (M3 Ultra). On held-out entities it beats the constant-velocity baseline
  **16.8px → 10.1px (−40%)** — a *learned* dynamics on Retina's structured state
  genuinely pays off. (Standalone torch script; no Retina dep on the Studio.)
- ⬜ push further: `TDMPC2Dynamics` (a famous engine) and the **latent channel**
  (feed V-JEPA per-entity vectors, not just positions) — both on the Mac Studio.
