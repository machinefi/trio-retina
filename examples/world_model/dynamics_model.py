"""Latent dynamics model — a small transformer world model over Retina states.

This is the Dreamer-4-aligned back-end (see README): a **transformer** trained
purely on **offline** recorded `WorldState` sequences, doing **imagination**
rollouts inside the learned model. The trio-retina twist: the dynamics consumes
Retina's *standardized, model-agnostic state* — symbolic position (from the
bbox) plus the appearance latent `vec` — instead of a baked-in tokenizer. The
state is the swappable interface.

Architecture (small on purpose; the point is the seam + the ablation, not scale):

  * One **token per (entity, timestep)** over a window of `K` past frames and
    `N` entity slots. Token features = motion `[cx, cy, vx, vy]` (normalized),
    optionally concatenated with a linear projection of the appearance `vec`.
  * Learned **temporal** and **entity-slot** positional embeddings are added.
  * A few `TransformerEncoder` layers with self-attention over ALL entity×time
    tokens — so the model attends over entities *and* their appearance vecs and
    can let appearance condition the predicted motion.
  * A per-entity head reads each entity's **last-timestep** token and predicts a
    **delta** `(dx, dy)` — the next-step centroid displacement. Predicting the
    residual makes the constant-velocity prior trivial to represent and learn.

The ablation lives in one flag: `with_appearance`. `True` includes the appearance
projection in the token; `False` zeroes it (pos-only). Same model, same training,
same everything else — so any difference is attributable to appearance.

`rollout()` does **imagination**: from a seed window it autoregressively predicts
the next state, appends it, slides the window, and repeats N steps — generating a
future trajectory entirely inside the learned model, in the standardized state
space.

All torch lives here (lazily, inside functions / a builder), so `import retina`
stays numpy-free. Install torch via the `[dynamics]` extra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Fixed entity-slot capacity for this 2-object scene. Slots are assigned by a
# stable sort of entity ids so the same object lands in the same slot each frame.
N_SLOTS = 2


@dataclass
class Normalizer:
    """Scales positions to ~unit range for stable training.

    Positions are divided by frame size; velocities are divided by `vscale`
    (a typical px/frame speed) inline in `windows_from_sequences`. Stored with
    the checkpoint so eval/rollout un-normalize consistently."""

    w: float
    h: float
    vscale: float = 8.0  # px/frame; covers the fast 'light' object's speed

    def pos(self, cx: float, cy: float) -> tuple[float, float]:
        return (cx / self.w, cy / self.h)

    def unpos(self, x: float, y: float) -> tuple[float, float]:
        return (x * self.w, y * self.h)


# ---------------------------------------------------------------------------
# Turning recorded sequences into supervised (window -> next) training tensors.
# numpy-only so the dataset prep needs no torch.
# ---------------------------------------------------------------------------


def _slot_index(eid: str, slot_map: dict[str, int]) -> int:
    """Assign a stable slot 0..N_SLOTS-1 to an entity id (first-seen order)."""
    if eid not in slot_map:
        slot_map[eid] = len(slot_map) % N_SLOTS
    return slot_map[eid]


def windows_from_sequences(
    sequences: list[list[dict]],
    *,
    k: int,
    w: float,
    h: float,
    vec_dim: int,
    vscale: float = 8.0,
    horizon: int = 1,
):
    """Build sliding (window, target) samples from raw recorded sequences.

    Returns numpy arrays:
      feat   (S, K, N, 4)        motion features [cx,cy,vx,vy], normalized
      vecs   (S, K, N, vec_dim)  appearance vectors (zeros for empty slots)
      mask   (S, N)              1 if entity slot present in the window's last frame
      target (S, N, 2)           H-step-ahead normalized delta (dx,dy) per slot

    Velocity at frame i is (pos_i - pos_{i-1}); the first frame in a window uses
    a zero velocity. Targets are the normalized displacement from the window's
    last frame to the frame `horizon` steps later.

    `horizon > 1` is the regime where APPEARANCE earns its keep: a short window
    of positions pins down the *local* velocity (so 1-step is easy for everyone),
    but the *type* — heavy keeps curving, light keeps zigzagging — governs where
    the object is several steps out. Type is identifiable from appearance, so the
    appearance channel can predict the type-specific divergence that short
    kinematics cannot.
    """
    norm = Normalizer(w, h, vscale)
    F, V, M, T = [], [], [], []
    for seq in sequences:
        # index per-frame entities by id, and record per-id positions
        slot_map: dict[str, int] = {}
        # pre-assign slots in stable id order across the whole sequence
        all_ids = sorted({e["id"] for st in seq for e in st["entities"]})
        for eid in all_ids:
            _slot_index(eid, slot_map)

        # per-frame slot tables: pos[(frame, slot)] etc.
        frames = []
        for st in seq:
            slot_pos = [None] * N_SLOTS
            slot_vec = [None] * N_SLOTS
            for e in st["entities"]:
                s = slot_map.get(e["id"])
                if s is None or s >= N_SLOTS:
                    continue
                slot_pos[s] = (e["cx"], e["cy"])
                slot_vec[s] = e.get("vec")
            frames.append((slot_pos, slot_vec))

        L = len(frames)
        # need `horizon` frames after the window for the target
        for start in range(0, L - k - horizon + 1):
            win = frames[start : start + k]
            nxt = frames[start + k + horizon - 1]
            feat = np.zeros((k, N_SLOTS, 4), np.float32)
            vec = np.zeros((k, N_SLOTS, vec_dim), np.float32)
            for ti in range(k):
                slot_pos, slot_vec = win[ti]
                prev_pos = win[ti - 1][0] if ti > 0 else slot_pos
                for s in range(N_SLOTS):
                    p = slot_pos[s]
                    if p is None:
                        continue
                    nx, ny = norm.pos(p[0], p[1])
                    pp = prev_pos[s] if prev_pos[s] is not None else p
                    vx, vy = (p[0] - pp[0]) / vscale, (p[1] - pp[1]) / vscale
                    feat[ti, s] = (nx, ny, vx, vy)
                    if slot_vec[s] is not None:
                        v = np.asarray(slot_vec[s], np.float32)
                        if v.shape[0] == vec_dim:
                            vec[ti, s] = v
            # target = normalized delta from last window frame to next frame
            mask = np.zeros(N_SLOTS, np.float32)
            target = np.zeros((N_SLOTS, 2), np.float32)
            last_pos = win[-1][0]
            nxt_pos = nxt[0]
            for s in range(N_SLOTS):
                if last_pos[s] is not None and nxt_pos[s] is not None:
                    dx = (nxt_pos[s][0] - last_pos[s][0]) / w
                    dy = (nxt_pos[s][1] - last_pos[s][1]) / h
                    target[s] = (dx, dy)
                    mask[s] = 1.0
            F.append(feat)
            V.append(vec)
            M.append(mask)
            T.append(target)

    if not F:
        raise ValueError("no training windows produced — check k vs seq_len")
    return (
        np.stack(F),
        np.stack(V),
        np.stack(M),
        np.stack(T),
    )


# ---------------------------------------------------------------------------
# The transformer (torch). Built lazily so importing this module is cheap and
# `import retina` never sees torch.
# ---------------------------------------------------------------------------


def build_model(*, vec_dim: int, with_appearance: bool, k: int, d_model: int = 64,
                n_layers: int = 3, n_heads: int = 4):
    """Construct the transformer dynamics model. Returns an nn.Module.

    `with_appearance` toggles the appearance branch — that single flag IS the
    ablation. `pos_only` (False) means the appearance projection is omitted, so
    the model literally cannot see `vec`."""
    import torch
    import torch.nn as nn

    class DynamicsTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.with_appearance = with_appearance
            self.d_model = d_model
            self.k = k
            self.motion_proj = nn.Linear(4, d_model)
            if with_appearance:
                self.vec_proj = nn.Sequential(
                    nn.Linear(vec_dim, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
                )
            self.temporal_emb = nn.Parameter(torch.zeros(k, d_model))
            self.slot_emb = nn.Parameter(torch.zeros(N_SLOTS, d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=0.0, batch_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2)
            )
            nn.init.normal_(self.temporal_emb, std=0.02)
            nn.init.normal_(self.slot_emb, std=0.02)

        def forward(self, feat, vec):
            # feat: (B,K,N,4)  vec: (B,K,N,vec_dim)
            b, k, n, _ = feat.shape
            tok = self.motion_proj(feat)  # (B,K,N,d)
            if self.with_appearance:
                tok = tok + self.vec_proj(vec)
            tok = tok + self.temporal_emb[None, :, None, :]
            tok = tok + self.slot_emb[None, None, :, :]
            tok = tok.reshape(b, k * n, self.d_model)  # flatten time×slots
            enc = self.encoder(tok)
            enc = enc.reshape(b, k, n, self.d_model)
            last = enc[:, -1]  # (B,N,d) — read the most recent timestep per slot
            return self.head(last)  # (B,N,2) predicted normalized delta

    return DynamicsTransformer()


# ---------------------------------------------------------------------------
# Imagination rollout — autoregressive forward simulation in state space.
# ---------------------------------------------------------------------------


def rollout(model, seed_window, *, steps: int, w: float, h: float, vscale: float = 8.0):
    """Imagine `steps` future frames from a seed window, in the learned model.

    `seed_window` is (K, N, 2+) raw centroids per slot (numpy; nan for absent).
    Returns an array (steps, N, 2) of predicted raw centroids — the imagined
    future trajectory rolled entirely inside the learned dynamics."""
    import torch

    k = model.k
    # window holds raw positions; vecs are held fixed from the seed (appearance
    # is ~constant for an object, so we carry the last observed vec forward).
    pos = [list(seed_window["pos"][i]) for i in range(k)]  # each: list of (cx,cy)|None
    vecs = seed_window["vec"]  # (N, vec_dim) last-known appearance per slot
    n = len(pos[0])
    vec_dim = vecs.shape[1]

    out = np.full((steps, n, 2), np.nan, np.float32)
    model.eval()
    for step in range(steps):
        feat = np.zeros((1, k, n, 4), np.float32)
        vecarr = np.zeros((1, k, n, vec_dim), np.float32)
        for ti in range(k):
            for s in range(n):
                p = pos[ti][s]
                if p is None:
                    continue
                pp = pos[ti - 1][s] if ti > 0 and pos[ti - 1][s] is not None else p
                feat[0, ti, s] = (p[0] / w, p[1] / h, (p[0] - pp[0]) / vscale, (p[1] - pp[1]) / vscale)
                vecarr[0, ti, s] = vecs[s]
        with torch.no_grad():
            delta = model(torch.from_numpy(feat), torch.from_numpy(vecarr)).numpy()[0]  # (N,2)
        new_pos: list = [None] * n
        last = pos[-1]
        for s in range(n):
            if last[s] is None:
                continue
            ncx = last[s][0] + float(delta[s, 0]) * w
            ncy = last[s][1] + float(delta[s, 1]) * h
            new_pos[s] = (ncx, ncy)
            out[step, s] = (ncx, ncy)
        pos.append(new_pos)
        pos.pop(0)
    return out
