# World-model front/back-end benchmark (early / illustrative)

A tiny grid over a world model's **dynamics input** × **prediction horizon**,
measuring held-out next-position error. It is the seed of a "decompose the
world model into a front end (any encoder) and a back end (any dynamics) that
meet on one standardized state, then grid over each" benchmark — NOT a published
result.

## Result (held-out position error, px — lower is better)

| dynamics input | horizon 3 | horizon 7 |
|---|---|---|
| const-velocity | 1.73 px | 7.68 px |
| pos-only | 0.49 px | 1.45 px |
| **pos+appearance** | 0.61 px | 1.33 px |

Mean over 3 sequence-level train/test splits (split by sequence, so no
window leaks across train/test). Run on `mps`.

**Finding.** Adding Retina's appearance latent (`entity.vec`, a frozen DINOv2
embedding) to the dynamics input lowers multi-step position error, and the gap
widens with the horizon. At horizon 7: pos+appearance beats constant-velocity
by **83%** and beats pos-only by **8%**. The
mechanism is honest and built into the scene: two appearance-distinct object
types follow different non-linear motion laws, so a short window of positions
fixes the *local* velocity (1-step is easy for everyone) but the *type* — only
legible from appearance — governs where the object is several steps out.

## Methodology (1 paragraph)

The synthetic scene has two visually-distinct types ("heavy" = large dark-red
block on a near-straight drift; "light" = small bright-cyan bullseye on a hard
banked turn) sharing the same instantaneous speed range, so velocity is not
type-diagnostic but appearance is. Each frame is run through the real Retina
pipeline (detector → IoU tracker → `dinov2-small` embedder → `WorldState`); the
dynamics is a small transformer (one token per entity×timestep, self-attention
over all tokens, a per-entity delta head) trained offline to predict the
H-step-ahead centroid displacement. The only thing that changes across the
learned rows is whether the appearance projection is included in the token —
that single flag IS the ablation; const-velocity is a model-free extrapolation
of the last observed velocity over the same windows.

## Caveats

- **Synthetic scene, small PoC.** 30 sequences, a tiny model
  (d_model=64, 2-frame window), `dinov2-small` vecs (dim 384).
- **Horizon-dependent.** At a SHORT horizon, positions alone already pin the
  motion, so pos-only can match or beat pos+appearance — appearance earns its
  keep only as the horizon grows and type (not local velocity) decides the
  future. The benchmark shows both regimes on purpose.
- **MPS run-to-run variance.** Numbers shift a little between runs; the robust
  signal is the *trend* (appearance's edge widens with the horizon), not the
  exact pixels.
- **Illustrative, not a claim.** Reproduce with the commands at the top of
  `examples/world_model/benchmark.py`.
