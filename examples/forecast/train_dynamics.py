"""Train a learned dynamics on Retina's WorldState trajectories (torch + MPS).

Standalone — needs only torch + numpy (no Retina), so it runs anywhere with torch.
Loads the JSON exported by export_trajectories.py, learns to
predict each entity's near-future displacement from a short window of its track,
and compares to the constant-velocity baseline on HELD-OUT entities. If the learned
model beats the baseline, a dynamics model on Retina's structured state is worth it.

    python3 train_dynamics.py /tmp/retina_traj.json
"""

import json
import random
import sys

import numpy as np
import torch
import torch.nn as nn

W, H, EPOCHS = 5, 5, 600  # window, horizon (frames @ export fps), training epochs


def load(path):
    d = json.load(open(path))
    wh = d["wh"]
    samples = []  # (entity_id, rel_window[W,2], target_delta[2], cur_vel[2])
    for eid, pts in d["traj"].items():
        p = np.array([[x[1] / wh[0], x[2] / wh[1]] for x in pts], dtype=np.float32)
        for i in range(W - 1, len(p) - H):
            samples.append((eid, p[i - W + 1 : i + 1] - p[i], p[i + H] - p[i], p[i] - p[i - 1]))
    return samples, wh


def split(samples, frac=0.8, seed=0):
    ids = sorted({s[0] for s in samples})
    random.Random(seed).shuffle(ids)
    train_ids = set(ids[: int(len(ids) * frac)])
    tr = [s for s in samples if s[0] in train_ids]
    te = [s for s in samples if s[0] not in train_ids]
    return tr, te


def tensors(samples, dev):
    return (
        torch.tensor(np.stack([s[1] for s in samples])).to(dev),
        torch.tensor(np.stack([s[2] for s in samples])).to(dev),
        torch.tensor(np.stack([s[3] for s in samples])).to(dev),
    )


def mae_px(pred, target, wh):
    d = (pred - target).detach().clone()
    d[:, 0] *= wh[0]
    d[:, 1] *= wh[1]
    return d.pow(2).sum(1).sqrt().mean().item()


class MLP(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(w * 2, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x.flatten(1))


def main(path, save=None):
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    samples, wh = load(path)
    tr, te = split(samples)
    Xtr, Ytr, _ = tensors(tr, dev)
    Xte, Yte, Vte = tensors(te, dev)

    base = mae_px(Vte * H, Yte, wh)  # constant velocity on the same test set

    model = MLP(W).to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    lossf = nn.MSELoss()
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        lossf(model(Xtr), Ytr).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        learned = mae_px(model(Xte), Yte, wh)

    print(f"device {dev}   train {len(tr)} / test {len(te)} samples   horizon {H} frames")
    print(f"  constant-velocity baseline:  {base:.1f} px")
    print(f"  learned dynamics (MLP):       {learned:.1f} px")
    if learned < base:
        print(f"\n→ learned dynamics beats velocity by {100*(1-learned/base):.0f}% on UNSEEN "
              "entities.\n  A dynamics model on Retina's structured state pays off.")
    else:
        print("\n→ no win here — try more data / a sequence model / the latent channel.")

    if save:
        torch.save({"sd": model.cpu().state_dict(), "W": W, "H": H, "wh": wh}, save)
        print(f"saved weights → {save}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/retina_traj.json",
         sys.argv[2] if len(sys.argv) > 2 else None)
