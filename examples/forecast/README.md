# Forecast — a dynamics model on top of Retina

The demo behind the hero GIF. It runs a **dynamics model** on Retina's
`WorldState` stream and predicts where each entity is headed (~1 s ahead) —
showing *why* Retina matters: a dynamics model consumes structured **state**, not
pixels, so Retina is the interface between any backbone and any forecaster.

**One state, two dynamics models.** The same `WorldState` feeds two forecasters
behind one tiny `DynamicsModel` protocol (`observe` / `predict`):

- `LinearForecaster` — constant-velocity baseline (gray arrow).
- `LearnedForecaster` — a small trained MLP (magenta arrow), **−35 %** centroid
  error vs the baseline on held-out entities.

Swap the detector (YOLO → V-JEPA → DINO) or the forecaster — the state in the
middle is the constant.

## Run it

```bash
python examples/forecast/quick_forecast.py     # synthetic — proves the loop, no model/GPU
python examples/forecast/forecast_video.py v.mp4   # real-video constant-velocity baseline
python examples/forecast/multi_consumer.py     # one WorldState -> rules + forecast + an LLM-judge stub
```

Reproduce the trained model and the annotated GIF:

```bash
python examples/forecast/export_trajectories.py v.mp4   # WorldState tracks -> JSON
python examples/forecast/train_dynamics.py traj.json dynamics.ckpt   # train the MLP (torch)
python examples/forecast/render_demo.py v.mp4 out.mp4   # the two-arrow demo video
```

## Files

| File | Role |
|------|------|
| `dynamics.py` | The `DynamicsModel` protocol + `LinearForecaster`, `LearnedForecaster`, and a `forecast_error` metric. |
| `train_dynamics.py` | Train the learned dynamics (standalone torch + numpy) on exported tracks. |
| `export_trajectories.py` | Run Retina (YOLO) → per-entity centroid tracks → JSON. |
| `render_demo.py` | Render the annotated two-arrow demo video. |
| `quick_forecast.py` · `forecast_video.py` · `multi_consumer.py` | Synthetic proof · real-video baseline · one-state-many-consumers. |
| `dynamics.ckpt` | The trained MLP weights (committed, so the GIF is reproducible). |

## What this is (and isn't)

This is a **demo of the next layer up**, not part of the Retina core — the core
stays the model-agnostic state layer. The forecaster here is deliberately small;
the point is the **seam**, not the model. A larger interaction-aware engine drops
in behind the same `DynamicsModel` protocol, and the **latent channel** (feeding
per-entity embeddings, not just positions) is the natural next step.
