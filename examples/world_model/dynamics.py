"""Train + eval the latent dynamics model — the HONEST appearance ablation.

This is the demo that answers the Phase-2 question: does Retina's **appearance
latent** (`entity.vec`, a frozen DINOv2 embedding) actually help a learned
dynamics model predict future positions, beyond what symbolic position alone
gives? We measure it on HELD-OUT sequences and report the real numbers — win,
tie, or loss.

Three predictors, one held-out test set, one metric (mean next-step centroid
error in **pixels**):

  1. constant-velocity baseline  — extrapolate last observed velocity (no model).
  2. learned  pos_only           — the transformer, appearance branch OFF.
  3. learned  with_appearance    — the transformer, appearance branch ON.

If (3) < (2) < (1), the latent channel earns its keep. We also run an
**imagination rollout**: from a seed window the with-appearance model imagines N
steps forward, and we compare the imagined trajectory to ground truth.

Everything torch lives here / in `dynamics_model.py` (lazy); `import retina`
stays numpy-free. Run on a machine with the `[dynamics]` extra:

    python examples/world_model/dataset.py --out examples/world_model/data/sequences.json
    python examples/world_model/dynamics.py --data examples/world_model/data/sequences.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from dynamics_model import (
    N_SLOTS,
    Normalizer,
    build_model,
    rollout,
    windows_from_sequences,
)

# ---------------------------------------------------------------------------
# Baselines / metrics (numpy-only).
# ---------------------------------------------------------------------------


def constant_velocity_error(sequences, k: int, w: float, h: float, horizon: int = 1) -> float:
    """Mean H-step-ahead centroid error (px) of a per-entity constant-velocity
    extrapolation, over the SAME (window -> target) samples the model is scored on.

    For each window, predict pos H steps out = last_pos + horizon*(last_pos - prev_pos)."""
    errs = []
    for seq in sequences:
        ids = sorted({e["id"] for st in seq for e in st["entities"]})
        slot = {eid: i % N_SLOTS for i, eid in enumerate(ids)}
        frames = []
        for st in seq:
            pos = [None] * N_SLOTS
            for e in st["entities"]:
                s = slot.get(e["id"])
                if s is not None and s < N_SLOTS:
                    pos[s] = (e["cx"], e["cy"])
            frames.append(pos)
        for start in range(0, len(frames) - k - horizon + 1):
            win = frames[start : start + k]
            nxt = frames[start + k + horizon - 1]
            last, prev = win[-1], win[-2] if k >= 2 else win[-1]
            for s in range(N_SLOTS):
                if last[s] is None or nxt[s] is None:
                    continue
                if prev[s] is not None:
                    vx, vy = last[s][0] - prev[s][0], last[s][1] - prev[s][1]
                else:
                    vx = vy = 0.0
                px, py = last[s][0] + vx * horizon, last[s][1] + vy * horizon
                errs.append(((px - nxt[s][0]) ** 2 + (py - nxt[s][1]) ** 2) ** 0.5)
    return float(np.mean(errs))


def model_test_error(model, feat, vec, mask, target, w: float, h: float) -> float:
    """Mean next-step centroid error (px) of the model over the test windows."""
    import torch

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(feat), torch.from_numpy(vec)).numpy()  # (S,N,2)
    # un-normalize deltas to pixels and measure displacement error vs target
    dpx = (pred[..., 0] - target[..., 0]) * w
    dpy = (pred[..., 1] - target[..., 1]) * h
    err = np.sqrt(dpx**2 + dpy**2)  # (S,N)
    m = mask > 0
    return float(err[m].mean())


# ---------------------------------------------------------------------------
# Training.
# ---------------------------------------------------------------------------


def train(model, feat, vec, mask, target, *, epochs: int, lr: float, device: str,
          batch: int = 256, seed: int = 0):
    import torch

    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ft = torch.from_numpy(feat).to(device)
    vt = torch.from_numpy(vec).to(device)
    mt = torch.from_numpy(mask).to(device)
    tt = torch.from_numpy(target).to(device)
    n = ft.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            pred = model(ft[idx], vt[idx])  # (B,N,2)
            diff = (pred - tt[idx]) ** 2  # (B,N,2)
            m = mt[idx][..., None]  # (B,N,1)
            loss = (diff * m).sum() / (m.sum() * 2 + 1e-9)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if ep == 0 or (ep + 1) % max(1, epochs // 5) == 0:
            print(f"      epoch {ep + 1:>3}/{epochs}  train_loss={tot / n:.6f}")
    return model.cpu()


# ---------------------------------------------------------------------------
# Imagination rollout vs ground truth.
# ---------------------------------------------------------------------------


def imagine_and_score(model, seq, *, k: int, steps: int, w: float, h: float,
                      vec_dim: int):
    """Seed the model with the first K frames of `seq`, imagine `steps` ahead,
    and compare to the real frames K..K+steps. Returns (mae_px, imagined, truth)."""
    ids = sorted({e["id"] for st in seq for e in st["entities"]})
    slot = {eid: i % N_SLOTS for i, eid in enumerate(ids)}
    pos_table = []
    last_vec = [np.zeros(vec_dim, np.float32) for _ in range(N_SLOTS)]
    for st in seq:
        pos = [None] * N_SLOTS
        for e in st["entities"]:
            s = slot.get(e["id"])
            if s is None or s >= N_SLOTS:
                continue
            pos[s] = (e["cx"], e["cy"])
            if e.get("vec") is not None:
                last_vec[s] = np.asarray(e["vec"], np.float32)
        pos_table.append(pos)

    seed = {
        "pos": [pos_table[i] for i in range(k)],
        "vec": np.stack(last_vec),
    }
    imagined = rollout(model, seed, steps=steps, w=w, h=h)  # (steps,N,2)

    # ground truth frames k..k+steps
    errs = []
    truth = np.full((steps, N_SLOTS, 2), np.nan, np.float32)
    for st_i in range(steps):
        fi = k + st_i
        if fi >= len(pos_table):
            break
        for s in range(N_SLOTS):
            gt = pos_table[fi][s]
            if gt is None or np.isnan(imagined[st_i, s, 0]):
                continue
            truth[st_i, s] = gt
            errs.append(
                ((imagined[st_i, s, 0] - gt[0]) ** 2 + (imagined[st_i, s, 1] - gt[1]) ** 2) ** 0.5
            )
    mae = float(np.mean(errs)) if errs else float("nan")
    return mae, imagined, truth


def save_rollout_png(imagined, truth, path: str, w: float, h: float) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = ["#d24", "#08c"]
    names = ["slot0", "slot1"]
    for s in range(imagined.shape[1]):
        ti = truth[:, s]
        im = imagined[:, s]
        good_t = ~np.isnan(ti[:, 0])
        good_i = ~np.isnan(im[:, 0])
        ax.plot(ti[good_t, 0], ti[good_t, 1], "-o", color=colors[s], label=f"{names[s]} truth", ms=3)
        ax.plot(im[good_i, 0], im[good_i, 1], "--x", color=colors[s], label=f"{names[s]} imagined", ms=4)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_title("Imagination rollout vs ground truth")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="examples/world_model/data/sequences.json")
    ap.add_argument(
        "--k",
        type=int,
        default=2,
        help="past-window length (short: too few frames to read curvature, so the "
        "type — and thus the future — is only legible from appearance)",
    )
    ap.add_argument(
        "--horizon",
        type=int,
        default=7,
        help="how many steps ahead to predict (>1 is where appearance earns its keep)",
    )
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rollout-steps", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--png", default="examples/world_model/media/rollout.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--splits",
        type=int,
        default=3,
        help="number of random train/test splits to average over (an honest, "
        "split-robust estimate — a one-split number is noisy on a small set)",
    )
    args = ap.parse_args()

    with open(args.data) as fp:
        data = json.load(fp)
    sequences = data["sequences"]
    w, h, vec_dim = float(data["W"]), float(data["H"]), int(data["vec_dim"])
    print(f"dataset: {len(sequences)} sequences, seq_len={data['seq_len']}, "
          f"vec={data['vec_model']} (dim {vec_dim})\n")

    device = args.device
    if device == "auto":
        import torch

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    print(f"device: {device}; window k={args.k}, horizon={args.horizon}; "
          f"splits={args.splits}\n")

    norm = Normalizer(w, h)
    last = {}  # carry the last split's with-appearance model + test seqs for rollout
    rows = {"cv": [], "pos": [], "app": []}
    for split in range(args.splits):
        seed = args.seed + split
        # split by SEQUENCE (no window leakage across train/test)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(sequences))
        n_test = max(1, int(len(sequences) * 0.2))
        test_idx = set(order[:n_test].tolist())
        train_seqs = [sequences[i] for i in range(len(sequences)) if i not in test_idx]
        test_seqs = [sequences[i] for i in sorted(test_idx)]

        tr = windows_from_sequences(
            train_seqs, k=args.k, w=w, h=h, vec_dim=vec_dim, vscale=norm.vscale, horizon=args.horizon
        )
        te = windows_from_sequences(
            test_seqs, k=args.k, w=w, h=h, vec_dim=vec_dim, vscale=norm.vscale, horizon=args.horizon
        )
        print(f"--- split {split + 1}/{args.splits} (seed {seed}): "
              f"train={len(train_seqs)} test={len(test_seqs)} seqs, "
              f"windows train={tr[0].shape[0]} test={te[0].shape[0]} ---")

        cv_err = constant_velocity_error(test_seqs, args.k, w, h, horizon=args.horizon)

        m_pos = build_model(vec_dim=vec_dim, with_appearance=False, k=args.k)
        m_pos = train(m_pos, *tr, epochs=args.epochs, lr=args.lr, device=device, seed=seed)
        pos_err = model_test_error(m_pos, *te, w, h)

        m_app = build_model(vec_dim=vec_dim, with_appearance=True, k=args.k)
        m_app = train(m_app, *tr, epochs=args.epochs, lr=args.lr, device=device, seed=seed)
        app_err = model_test_error(m_app, *te, w, h)

        rows["cv"].append(cv_err)
        rows["pos"].append(pos_err)
        rows["app"].append(app_err)
        print(f"    cv={cv_err:.3f}  pos_only={pos_err:.3f}  with_appearance={app_err:.3f} px\n")
        last = {"train_seqs": train_seqs, "test_seqs": test_seqs}

    cv = float(np.mean(rows["cv"]))
    pos = float(np.mean(rows["pos"]))
    app = float(np.mean(rows["app"]))

    # --- results table (averaged over splits) ---
    print("=" * 60)
    print(f"HELD-OUT {args.horizon}-STEP POSITION ERROR — mean over {args.splits} "
          "splits (px, lower=better)")
    print("=" * 60)
    print(f"  {'constant-velocity baseline':<32} {cv:8.3f} px")
    print(f"  {'learned  pos_only':<32} {pos:8.3f} px")
    print(f"  {'learned  with_appearance':<32} {app:8.3f} px")
    print("=" * 60)
    impr_vs_cv = (cv - app) / cv * 100
    impr_app = (pos - app) / pos * 100
    print(f"  with_appearance vs constant-velocity : {impr_vs_cv:+.1f}%")
    print(f"  with_appearance vs pos_only          : {impr_app:+.1f}%")
    # how many splits did appearance win on (honest spread)
    wins = sum(1 for a, p in zip(rows["app"], rows["pos"], strict=True) if a < p)
    print(f"  appearance beat pos_only on {wins}/{args.splits} splits")
    if app < pos and pos < cv:
        print("  VERDICT: appearance HELPS (and both beat constant-velocity).")
    elif app < pos:
        print("  VERDICT: appearance helps over pos_only.")
    elif pos < cv:
        print("  VERDICT: learned beats constant-velocity, but appearance did NOT help.")
    else:
        print("  VERDICT: learned did not beat constant-velocity on this run.")
    print("=" * 60)

    # --- imagination rollout ---
    # The ablation above trains a HORIZON-step predictor (that's where appearance
    # earns its keep). Autoregressive imagination, however, advances ONE frame at
    # a time, so it needs a 1-step model — feeding a 7-step delta into a 1-frame
    # update would fly the trajectory off the frame. So we train a dedicated
    # 1-step with-appearance model on the last split's training sequences and roll
    # it out autoregressively on a held-out sequence.
    roll_tr = windows_from_sequences(
        last["train_seqs"], k=args.k, w=w, h=h, vec_dim=vec_dim,
        vscale=norm.vscale, horizon=1,
    )
    m_roll = build_model(vec_dim=vec_dim, with_appearance=True, k=args.k)
    print("\ntraining 1-step model for imagination rollout…")
    m_roll = train(m_roll, *roll_tr, epochs=args.epochs, lr=args.lr,
                   device=device, seed=args.seed)

    long_seq = max(last["test_seqs"], key=len)
    steps = min(args.rollout_steps, len(long_seq) - args.k)
    mae, imagined, truth = imagine_and_score(
        m_roll, long_seq, k=args.k, steps=steps, w=w, h=h, vec_dim=vec_dim
    )
    print(f"\nimagination rollout ({steps} steps, 1-step with_appearance, held-out seq):")
    print(f"  mean per-frame trajectory error vs ground truth : {mae:.3f} px")

    if args.png:
        import os

        os.makedirs(os.path.dirname(args.png) or ".", exist_ok=True)
        if save_rollout_png(imagined, truth, args.png, w, h):
            print(f"  saved rollout visualization → {args.png}")


if __name__ == "__main__":
    main()
