"""A small benchmark grid — the seed of a world-model front/back-end benchmark.

The idea Phase 2 demonstrated: decompose a world model into a *front end* (any
perception encoder) and a *back end* (any dynamics) that meet on ONE
standardized state (Retina's `WorldState`). Once they're decoupled you can hold
one fixed and grid over the other — exactly what a benchmark does.

This script runs a deliberately small grid:

    dynamics input  ∈ {constant-velocity, pos-only, pos+appearance}
                  ×
    prediction horizon ∈ {a short one, a longer one}

and prints a clean held-out position-error table (px, lower = better), averaged
over a few train/test splits (split by SEQUENCE — no window leakage). It writes
the same table to `BENCHMARK.md` so the result is captured, not just printed.

This is EARLY / ILLUSTRATIVE, not a published claim: a synthetic scene, a tiny
model, an MPS run with real run-to-run variance. The honest finding it
surfaces is narrow and real: on a scene where the motion law is type-dependent
and the type is only legible from appearance, adding the appearance latent to
the dynamics input measurably lowers multi-step position error — and the gap
WIDENS with the horizon, because that's where short kinematics run out and type
(hence appearance) starts to matter.

Reuses the Phase-2 back end (import, don't duplicate). Run on a torch machine:

    pip install 'trio-retina[dynamics]'
    python examples/world_model/dataset.py --n 12 --len 24 \
        --out examples/world_model/data/sequences.json
    python examples/world_model/benchmark.py \
        --data examples/world_model/data/sequences.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamics import (  # noqa: E402
    constant_velocity_error,
    model_test_error,
    train,
)
from dynamics_model import build_model, windows_from_sequences  # noqa: E402

# The grid's row order / display names.
INPUTS = ["const-velocity", "pos-only", "pos+appearance"]


def _pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _split(sequences: list, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sequences))
    n_test = max(1, int(len(sequences) * 0.2))
    test = set(order[:n_test].tolist())
    train = [sequences[i] for i in range(len(sequences)) if i not in test]
    held = [sequences[i] for i in sorted(test)]
    return train, held


def _eval_cell(
    input_kind: str,
    train_seqs: list,
    test_seqs: list,
    *,
    k: int,
    horizon: int,
    w: float,
    h: float,
    vec_dim: int,
    epochs: int,
    lr: float,
    device: str,
    seed: int,
) -> float:
    """Held-out position error (px) for one (input × horizon) cell, one split."""
    if input_kind == "const-velocity":
        return constant_velocity_error(test_seqs, k, w, h, horizon=horizon)
    with_appearance = input_kind == "pos+appearance"
    tr = windows_from_sequences(
        train_seqs, k=k, w=w, h=h, vec_dim=vec_dim, horizon=horizon
    )
    te = windows_from_sequences(
        test_seqs, k=k, w=w, h=h, vec_dim=vec_dim, horizon=horizon
    )
    model = build_model(vec_dim=vec_dim, with_appearance=with_appearance, k=k)
    model = train(model, *tr, epochs=epochs, lr=lr, device=device, seed=seed)
    return model_test_error(model, *te, w, h)


def run_grid(
    sequences: list,
    *,
    horizons: list[int],
    k: int,
    w: float,
    h: float,
    vec_dim: int,
    epochs: int,
    lr: float,
    device: str,
    splits: int,
    seed: int,
) -> dict[tuple[str, int], float]:
    """Run the full grid; return {(input_kind, horizon): mean_px_over_splits}."""
    results: dict[tuple[str, int], float] = {}
    for horizon in horizons:
        for input_kind in INPUTS:
            per_split = []
            for split in range(splits):
                s = seed + split
                train_seqs, test_seqs = _split(sequences, s)
                err = _eval_cell(
                    input_kind, train_seqs, test_seqs,
                    k=k, horizon=horizon, w=w, h=h, vec_dim=vec_dim,
                    epochs=epochs, lr=lr, device=device, seed=s,
                )
                per_split.append(err)
            results[(input_kind, horizon)] = float(np.mean(per_split))
            print(
                f"  horizon={horizon:>2}  {input_kind:<16} "
                f"= {results[(input_kind, horizon)]:7.3f} px  "
                f"(mean of {splits} splits)"
            )
    return results


def render_table(results: dict[tuple[str, int], float], horizons: list[int]) -> str:
    """Markdown table: rows = dynamics input, cols = horizon (px error)."""
    head = "| dynamics input | " + " | ".join(f"horizon {hz}" for hz in horizons) + " |"
    sep = "|---|" + "|".join(["---"] * len(horizons)) + "|"
    lines = [head, sep]
    for input_kind in INPUTS:
        cells = [f"{results[(input_kind, hz)]:.2f} px" for hz in horizons]
        label = f"**{input_kind}**" if input_kind == "pos+appearance" else input_kind
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_benchmark_md(
    path: str,
    results: dict[tuple[str, int], float],
    horizons: list[int],
    *,
    sequences: int,
    vec_model: str,
    vec_dim: int,
    device: str,
    splits: int,
    k: int,
) -> None:
    table = render_table(results, horizons)
    # the headline improvement at the LONGEST horizon (where appearance matters)
    hz = horizons[-1]
    cv = results[("const-velocity", hz)]
    pos = results[("pos-only", hz)]
    app = results[("pos+appearance", hz)]
    vs_cv = (cv - app) / cv * 100 if cv else 0.0
    vs_pos = (pos - app) / pos * 100 if pos else 0.0

    md = f"""# World-model front/back-end benchmark (early / illustrative)

A tiny grid over a world model's **dynamics input** × **prediction horizon**,
measuring held-out next-position error. It is the seed of a "decompose the
world model into a front end (any encoder) and a back end (any dynamics) that
meet on one standardized state, then grid over each" benchmark — NOT a published
result.

## Result (held-out position error, px — lower is better)

{table}

Mean over {splits} sequence-level train/test splits (split by sequence, so no
window leaks across train/test). Run on `{device}`.

**Finding.** Adding Retina's appearance latent (`entity.vec`, a frozen DINOv2
embedding) to the dynamics input lowers multi-step position error, and the gap
widens with the horizon. At horizon {hz}: pos+appearance beats constant-velocity
by **{vs_cv:.0f}%** and beats pos-only by **{vs_pos:.0f}%**. The
mechanism is honest and built into the scene: two appearance-distinct object
types follow different non-linear motion laws, so a short window of positions
fixes the *local* velocity (1-step is easy for everyone) but the *type* — only
legible from appearance — governs where the object is several steps out.

## Methodology (1 paragraph)

The synthetic scene has two visually-distinct types ("heavy" = large dark-red
block on a near-straight drift; "light" = small bright-cyan bullseye on a hard
banked turn) sharing the same instantaneous speed range, so velocity is not
type-diagnostic but appearance is. Each frame is run through the real Retina
pipeline (detector → IoU tracker → `{vec_model}` embedder → `WorldState`); the
dynamics is a small transformer (one token per entity×timestep, self-attention
over all tokens, a per-entity delta head) trained offline to predict the
H-step-ahead centroid displacement. The only thing that changes across the
learned rows is whether the appearance projection is included in the token —
that single flag IS the ablation; const-velocity is a model-free extrapolation
of the last observed velocity over the same windows.

## Caveats

- **Synthetic scene, small PoC.** {sequences} sequences, a tiny model
  (d_model=64, {k}-frame window), `{vec_model}` vecs (dim {vec_dim}).
- **Horizon-dependent.** At a SHORT horizon, positions alone already pin the
  motion, so pos-only can match or beat pos+appearance — appearance earns its
  keep only as the horizon grows and type (not local velocity) decides the
  future. The benchmark shows both regimes on purpose.
- **MPS run-to-run variance.** Numbers shift a little between runs; the robust
  signal is the *trend* (appearance's edge widens with the horizon), not the
  exact pixels.
- **Illustrative, not a claim.** Reproduce with the commands at the top of
  `examples/world_model/benchmark.py`.
"""
    Path(path).write_text(md)
    print(f"\nwrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="World-model front/back-end benchmark grid.")
    ap.add_argument("--data", default="examples/world_model/data/sequences.json")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--horizons", type=int, nargs="+", default=[3, 7])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--splits", type=int, default=3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--md", default="BENCHMARK.md")
    args = ap.parse_args()

    with open(args.data) as fp:
        data = json.load(fp)
    sequences = data["sequences"]
    w, h, vec_dim = float(data["W"]), float(data["H"]), int(data["vec_dim"])
    device = _pick_device(args.device)

    print("=" * 60)
    print("  WORLD-MODEL FRONT/BACK-END BENCHMARK GRID (small / illustrative)")
    print("=" * 60)
    print(
        f"dataset: {len(sequences)} seqs × {data['seq_len']} frames, "
        f"vec={data['vec_model']} (dim {vec_dim}); device={device}; "
        f"k={args.k}, horizons={args.horizons}, splits={args.splits}\n"
    )

    results = run_grid(
        sequences,
        horizons=args.horizons, k=args.k, w=w, h=h, vec_dim=vec_dim,
        epochs=args.epochs, lr=args.lr, device=device,
        splits=args.splits, seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("  HELD-OUT POSITION ERROR (px, lower = better)")
    print("=" * 60)
    print(render_table(results, args.horizons))
    print("=" * 60)

    write_benchmark_md(
        args.md, results, args.horizons,
        sequences=len(sequences), vec_model=data["vec_model"], vec_dim=vec_dim,
        device=device, splits=args.splits, k=args.k,
    )


if __name__ == "__main__":
    main()
