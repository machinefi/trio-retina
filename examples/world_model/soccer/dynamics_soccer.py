"""A small multi-player dynamics transformer over Retina soccer WorldStates.

Same idea as the synthetic-scene back-end (`examples/world_model/dynamics_model.py`)
— one token per (player, timestep), self-attention over all player×time tokens,
a per-player head that predicts the next-step displacement `(dx, dy)` — but
generalized from the 2-slot toy scene to **N players** with a presence mask, so it
can learn from real, partially-observed soccer tracks.

The token features are motion `[cx, cy, vx, vy]` (normalized) optionally
concatenated with a projection of the player's frozen DINOv2 appearance `vec`.
Attending over *all* players lets the model use the configuration of the whole
scene (teammates, opponents, where the play is) — exactly the structure a soccer
world model should exploit — when predicting any one player's next move.

Honesty note: player motion is stochastic (a player can cut, accelerate, or stop
at will), so even a good model predicts the *next short step* with real error.
We keep the horizon short and report the true held-out error; we do not pretend
the predictions are tight.

All torch is lazy / inside functions so the numpy-only `retina` core is untouched.
"""

from __future__ import annotations

import numpy as np


class Normalizer:
    """Scale pitch-relative positions to ~unit range for stable training."""

    def __init__(self, w: float, h: float, vscale: float = 20.0):
        self.w = w
        self.h = h
        self.vscale = vscale  # px/frame; a typical fast player stride at this fps


def build_windows(
    seq: list[dict],
    *,
    k: int,
    n_slots: int,
    w: float,
    h: float,
    vec_dim: int,
    vscale: float,
    horizon: int = 1,
    id_to_slot: dict[str, int] | None = None,
):
    """Turn one long soccer sequence into sliding (window -> H-step target) samples.

    `seq` is a list of frames; each frame is a list of `{slot, cx, cy, vec}` where
    `slot` is a stable 0..n_slots-1 index per player track (assigned upstream).

    Returns numpy arrays:
      feat   (S, K, N, 4)        motion [cx,cy,vx,vy] normalized
      vecs   (S, K, N, vec_dim)  appearance vectors (zeros where absent)
      mask   (S, N)              1 if the slot has BOTH a window-last and target pos
      target (S, N, 2)           H-step-ahead normalized delta (dx,dy)
    """
    norm = Normalizer(w, h, vscale)
    # frames as slot tables
    table = []
    for fr in seq:
        pos = [None] * n_slots
        vec = [None] * n_slots
        for e in fr:
            s = e["slot"]
            if 0 <= s < n_slots:
                pos[s] = (e["cx"], e["cy"])
                vec[s] = e.get("vec")
        table.append((pos, vec))

    F, V, M, T = [], [], [], []
    L = len(table)
    for start in range(0, L - k - horizon + 1):
        win = table[start : start + k]
        nxt_pos = table[start + k + horizon - 1][0]
        feat = np.zeros((k, n_slots, 4), np.float32)
        vecarr = np.zeros((k, n_slots, vec_dim), np.float32)
        for ti in range(k):
            slot_pos, slot_vec = win[ti]
            prev_pos = win[ti - 1][0] if ti > 0 else slot_pos
            for s in range(n_slots):
                p = slot_pos[s]
                if p is None:
                    continue
                nx, ny = p[0] / w, p[1] / h
                pp = prev_pos[s] if prev_pos[s] is not None else p
                vx, vy = (p[0] - pp[0]) / vscale, (p[1] - pp[1]) / vscale
                feat[ti, s] = (nx, ny, vx, vy)
                v = slot_vec[s]
                if v is not None and len(v) == vec_dim:
                    vecarr[ti, s] = np.asarray(v, np.float32)
        mask = np.zeros(n_slots, np.float32)
        target = np.zeros((n_slots, 2), np.float32)
        last_pos = win[-1][0]
        for s in range(n_slots):
            if last_pos[s] is not None and nxt_pos[s] is not None:
                target[s] = ((nxt_pos[s][0] - last_pos[s][0]) / w,
                             (nxt_pos[s][1] - last_pos[s][1]) / h)
                mask[s] = 1.0
        F.append(feat)
        V.append(vecarr)
        M.append(mask)
        T.append(target)

    if not F:
        raise ValueError("no training windows produced — check k/horizon vs seq length")
    _ = id_to_slot, norm  # kept for symmetry/readability
    return np.stack(F), np.stack(V), np.stack(M), np.stack(T)


def build_model(*, vec_dim: int, with_appearance: bool, k: int, n_slots: int,
                d_model: int = 96, n_layers: int = 3, n_heads: int = 4):
    """Construct the multi-player dynamics transformer (torch). `with_appearance`
    toggles the DINOv2 branch — that one flag is the ablation."""
    import torch
    import torch.nn as nn

    class SoccerDynamics(nn.Module):
        def __init__(self):
            super().__init__()
            self.with_appearance = with_appearance
            self.d_model = d_model
            self.k = k
            self.n_slots = n_slots
            self.motion_proj = nn.Linear(4, d_model)
            if with_appearance:
                self.vec_proj = nn.Sequential(
                    nn.Linear(vec_dim, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
                )
            self.temporal_emb = nn.Parameter(torch.zeros(k, d_model))
            self.slot_emb = nn.Parameter(torch.zeros(n_slots, d_model))
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
            b, k, n, _ = feat.shape
            tok = self.motion_proj(feat)
            if self.with_appearance:
                tok = tok + self.vec_proj(vec)
            tok = tok + self.temporal_emb[None, :, None, :]
            tok = tok + self.slot_emb[None, None, :, :]
            tok = tok.reshape(b, k * n, self.d_model)
            enc = self.encoder(tok)
            enc = enc.reshape(b, k, n, self.d_model)
            last = enc[:, -1]
            return self.head(last)

    return SoccerDynamics()


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
            pred = model(ft[idx], vt[idx])
            diff = (pred - tt[idx]) ** 2
            m = mt[idx][..., None]
            loss = (diff * m).sum() / (m.sum() * 2 + 1e-9)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if ep == 0 or (ep + 1) % max(1, epochs // 5) == 0:
            print(f"      epoch {ep + 1:>3}/{epochs}  train_loss={tot / n:.6f}")
    return model.cpu()


def predict_step(model, feat, vec):
    """One forward pass: (1,K,N,4)/(1,K,N,vec) -> (N,2) normalized deltas."""
    import torch

    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(feat), torch.from_numpy(vec)).numpy()[0]
