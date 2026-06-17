"""The world-model stack, end to end, in one runnable script.

This is the finale of the trio-retina world-model demo (Wow #3: a
pip-installable world-model stack). It wires the WHOLE thing on the synthetic
scene and narrates the journey:

    perception encoder            (a frozen DINOv2 fills entity.vec)
        │  s = Enc(x)
        ▼
    Retina WorldState             (symbolic core + the latent vec channel)
        │  one standardized, model-agnostic state
        ▼
    learned latent dynamics       (a small transformer trained offline)
        │  ŝ' = Dyn(s)
        ▼
    imagination rollout           (autoregress forward, INSIDE the model)

The point is the seam, not the scale: a front end (any encoder) and a back end
(any dynamics) compose through ONE standardized state. Swap either without
touching the other. Here we use real DINOv2 for the front and the Phase-2
transformer for the back, on a single held-out sequence, and print:

  1. a frame's WorldState (the standardized hand-off — symbolic + latent),
  2. the dynamics imagining N steps ahead, with the imagined trajectory vs the
     ground truth it never saw.

We REUSE the Phase-2 dataset/model code (import, don't duplicate). Run it with
the dynamics + dino extras on a machine with torch (Mac Studio / MPS / CUDA):

    pip install 'trio-retina[dynamics,dino]'
    # generate a tiny dataset with REAL DINOv2 appearance vecs:
    python examples/world_model/dataset.py --n 12 --len 24 \
        --out examples/world_model/data/sequences.json
    # then wire the full stack end to end on it:
    python examples/world_model/end_to_end.py \
        --data examples/world_model/data/sequences.json

Numpy-only? Pass a `--no-dino`-generated dataset; the script still runs the full
seam (deterministic stand-in vecs instead of DINOv2). The seam is identical —
only the encoder behind `entity.vec` differs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Reuse Phase-2's back-end code (sibling modules), without duplicating it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamics import imagine_and_score  # noqa: E402
from dynamics_model import build_model, windows_from_sequences  # noqa: E402


def _pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _print_worldstate(seq: list[dict], frame_idx: int, vec_model: str) -> None:
    """Print one frame's WorldState — the standardized front→back hand-off."""
    st = seq[frame_idx]
    print("─" * 64)
    print(f"① a Retina WorldState  (frame t={st['t']:.0f}; encoder={vec_model})")
    print("─" * 64)
    print("  the front end (perception encoder) produced ONE standardized state:")
    print("  symbolic core (id/type/bbox) + a model-tagged latent vec channel.\n")
    for e in st["entities"]:
        vec = e.get("vec") or []
        head = ", ".join(f"{v:+.3f}" for v in vec[:4])
        print(
            f"    entity id={e['id']!s:<3} type={e['type']:<6} "
            f"center=({e['cx']:6.1f},{e['cy']:6.1f})  "
            f"vec[{vec_model}, dim {len(vec)}]=[{head}, …]"
        )
    print(
        "\n  → this state is the swappable interface. The dynamics back end below\n"
        "    reads it WITHOUT knowing which encoder filled the latent."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="examples/world_model/data/sequences.json")
    ap.add_argument("--k", type=int, default=2, help="seed-window length")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rollout-steps", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.data) as fp:
        data = json.load(fp)
    sequences = data["sequences"]
    w, h, vec_dim = float(data["W"]), float(data["H"]), int(data["vec_dim"])
    vec_model = data["vec_model"]

    print("=" * 64)
    print("  THE WORLD-MODEL STACK, END TO END")
    print("  perception encoder → Retina WorldState → learned dynamics → dream")
    print("=" * 64)
    print(
        f"\ndataset: {len(sequences)} sequences × {data['seq_len']} frames; "
        f"appearance vec = {vec_model} (dim {vec_dim})"
    )

    # Hold out one sequence end to end; train the back end on the rest.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(sequences)).tolist()
    test_i = order[0]
    train_seqs = [sequences[i] for i in order[1:]]
    test_seq = sequences[test_i]
    print(f"held-out sequence (never trained on): index {test_i}\n")

    # ① show the standardized state the front end hands the back end.
    _print_worldstate(test_seq, args.k, vec_model)

    # ② train the back end (a small transformer) on the standardized states and
    #    let it IMAGINE the held-out future. 1-step model → autoregressive roll.
    device = _pick_device(args.device)
    print("\n" + "─" * 64)
    print(f"② a learned dynamics model imagines the future  (device={device})")
    print("─" * 64)
    tr = windows_from_sequences(
        train_seqs, k=args.k, w=w, h=h, vec_dim=vec_dim, horizon=1
    )
    model = build_model(vec_dim=vec_dim, with_appearance=True, k=args.k)
    print(
        f"  training a 1-step transformer (with appearance) on {len(train_seqs)} "
        f"held-out-excluded sequences …"
    )
    from dynamics import train  # noqa: E402  (lazy: torch only here)

    model = train(model, *tr, epochs=args.epochs, lr=args.lr, device=device, seed=args.seed)

    steps = min(args.rollout_steps, len(test_seq) - args.k)
    mae, imagined, truth = imagine_and_score(
        model, test_seq, k=args.k, steps=steps, w=w, h=h, vec_dim=vec_dim
    )
    print(
        f"\n  the dynamics imagines {steps} steps ahead off the seed window — "
        "autoregressively,\n  entirely INSIDE the learned model (it never sees "
        "the real future frames):\n"
    )
    print(f"    {'step':>4} {'slot':>4} {'imagined (cx,cy)':>22} {'truth (cx,cy)':>22} {'err px':>8}")
    for s_i in range(steps):
        for slot in range(imagined.shape[1]):
            im = imagined[s_i, slot]
            gt = truth[s_i, slot]
            if np.isnan(im[0]) or np.isnan(gt[0]):
                continue
            err = float(np.hypot(im[0] - gt[0], im[1] - gt[1]))
            print(
                f"    {s_i:>4} {slot:>4} "
                f"  ({im[0]:7.1f},{im[1]:7.1f})    "
                f"  ({gt[0]:7.1f},{gt[1]:7.1f})  {err:8.2f}"
            )
    print(f"\n  mean imagined-vs-truth trajectory error: {mae:.2f} px/frame")

    print("\n" + "=" * 64)
    print("  front (any encoder) + back (any dynamics) composed through ONE state.")
    print("  swap the encoder → examples/world_model/multi_encoder.py")
    print("  the full ablation grid → examples/world_model/benchmark.py")
    print("=" * 64)


if __name__ == "__main__":
    main()
