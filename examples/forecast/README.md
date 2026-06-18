# Forecast — a dynamics model on top of Retina

The demo behind the hero GIF. It runs a **dynamics model** on Retina's
`WorldState` stream and predicts where each entity is headed (~1 s ahead) —
showing *why* Retina matters: a dynamics model consumes structured **state**, not
pixels, so Retina is the interface between any backbone and any forecaster.

**One state, two dynamics models.** The same `WorldState` feeds two forecasters
behind one tiny `DynamicsModel` protocol (`observe` / `predict`):

- `LinearForecaster` — constant-velocity baseline (gray arrow).
- `LearnedForecaster` — a small trained MLP (magenta arrow), which *can* anticipate
  turns / slow-downs a constant-velocity model misses.

Swap the detector (YOLO → V-JEPA → DINO) or the forecaster — the state in the
middle is the constant.

## Run it — the reproducible demo (no model, no GPU, no footage)

```bash
python examples/forecast/quick_forecast.py
```

This is the runnable evidence: it builds a synthetic `WorldState` stream, forecasts
a few frames ahead, and prints the real baseline-vs-baseline comparison — a
constant-velocity model vs a no-motion model — so you can see the WorldState is
genuinely dynamics-ready. Whatever the numbers are, they come straight from
committed code.

### One state, many consumers — needs a clip ([video]+[yolo])

`multi_consumer.py` runs YOLO over a real video and fans the single `WorldState`
stream out to rules + forecast + an LLM-judge stub, so it needs `[video]`+`[yolo]`
installed and a clip of your own (defaults to `/tmp/demo.mp4`):

```bash
python examples/forecast/multi_consumer.py your_clip.mp4   # one WorldState -> rules + forecast + an LLM-judge stub
```

### The headline real-video result needs your own footage

The learned-vs-baseline comparison and the annotated GIF were measured on real
traffic video, which we don't redistribute — so that specific result isn't
reproducible from this repo alone (only the trained checkpoint is committed). A
learned MLP does **not** automatically beat constant velocity; `train_dynamics.py`
prints the honest win-or-tie on *your* data. To regenerate it end-to-end on a clip
of your own:

```bash
python examples/forecast/export_trajectories.py v.mp4 traj.json   # WorldState tracks -> JSON ([video]+[yolo])
python examples/forecast/train_dynamics.py traj.json dynamics.ckpt   # train + report baseline vs learned (torch)
python examples/forecast/forecast_video.py v.mp4                  # real-video constant-velocity baseline
python examples/forecast/render_demo.py v.mp4 out.mp4             # the two-arrow demo video
```

## Files

| File | Role |
|------|------|
| `dynamics.py` | The `DynamicsModel` protocol + `LinearForecaster`, `LearnedForecaster`, and a `forecast_error` metric. |
| `train_dynamics.py` | Train the learned dynamics (standalone torch + numpy) on exported tracks. |
| `export_trajectories.py` | Run Retina (YOLO) → per-entity centroid tracks → JSON. |
| `render_demo.py` | Render the annotated two-arrow demo video. |
| `quick_forecast.py` · `forecast_video.py` · `multi_consumer.py` | Synthetic proof · real-video baseline · one-state-many-consumers. |
| `dynamics.ckpt` | The trained MLP weights from the demo footage (committed for the GIF; retrain on your own clip with `train_dynamics.py`). |

## What this is (and isn't)

This is a **demo of the next layer up**, not part of the Retina core — the core
stays the model-agnostic state layer. The forecaster here is deliberately small;
the point is the **seam**, not the model. A larger interaction-aware engine drops
in behind the same `DynamicsModel` protocol, and the **latent channel** (feeding
per-entity embeddings, not just positions) is the natural next step.
