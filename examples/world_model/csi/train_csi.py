"""End-to-end CSI world model THROUGH Retina — train, ablate, imagine, evaluate.

Pipeline (all offline, synthetic CSI):

  1. DATA      synthesize CSI walks (csi_data) — moving subject + multipath.
  2. STATE     assemble a real retina.WorldState per timestep (csi_state) and
               serialize it, proving CSI rides Retina's dual-channel schema.
  3. ENCODE    CSI features -> latent z via the JEPA online encoder.
  4. DYNAMICS  action-conditioned JEPA: predict z_{t+1} from (z_t, a_t=velocity),
               homomorphic transition, latent-space loss + VICReg (csi_dynamics).
  5. FUSION/   ABLATION: action-conditioned (CSI + velocity action) vs CSI-only
     ABLATION  (action ignored). Lower next-latent error => the velocity channel
               (multimodal fusion) helps — analogous to the with_appearance flag.
  6. IMAGINE   roll the latent forward under an action sequence (no CSI after the
               seed) and read out POSITION via a frozen linear probe; beat a
               constant-velocity baseline on held-out walks.

Reports REAL numbers from an actual run. Tiny model, few epochs — must run e2e.

    python examples/world_model/csi/train_csi.py
"""

from __future__ import annotations

import argparse

import numpy as np

from csi_data import (
    RoomConfig,
    csi_to_features,
    feature_dim,
    make_csi_sequence,
)
from csi_dynamics import (
    CSIConfig,
    build_jepa,
    jepa_loss_multistep,
    rollout_latent,
)
from csi_state import csi_worldstate


def build_dataset(cfg: RoomConfig, *, n_seqs: int, n_steps: int, seed: int):
    """Make `n_seqs` CSI walks; return per-seq feature/action/pos arrays."""
    seqs = []
    for i in range(n_seqs):
        s = make_csi_sequence(cfg, n_steps=n_steps, seed=seed + i)
        feats = np.stack([csi_to_features(h) for h in s["H"]])  # (T, feat_dim)
        seqs.append({"feat": feats, "vel": s["vel"], "pos": s["pos"], "H": s["H"]})
    return seqs


def subrollout_windows(seqs, *, L):
    """Sliding windows of L consecutive CSI frames + their L-1 actions.

    Returns X (S, L, feat) and A (S, L-1, action). Training matches a predictor
    rolled L-1 steps to the true latents — the multi-step regime where the
    velocity action is necessary (1-step is trivially "predict z_t")."""
    Xw, Aw = [], []
    for s in seqs:
        f, v = s["feat"], s["vel"]
        T = len(f)
        for start in range(0, T - L + 1):
            Xw.append(f[start : start + L])
            Aw.append(v[start : start + L - 1])
    return np.stack(Xw), np.stack(Aw)


def train_jepa(model, Xw, Aw, *, epochs, lr, batch, device, seed=0):
    import torch

    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    Xt = torch.from_numpy(Xw).to(device)
    At = torch.from_numpy(Aw).to(device)
    n = Xt.shape[0]
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = {"pred": 0.0, "var": 0.0, "cov": 0.0}
        nb = 0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            loss, parts = jepa_loss_multistep(model, Xt[idx], At[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.update_target(tau=0.99)
            for k in tot:
                tot[k] += parts[k]
            nb += 1
        for k in tot:
            tot[k] /= max(nb, 1)
        hist.append(tot)
        if ep == 0 or (ep + 1) % max(1, epochs // 5) == 0:
            print(f"      epoch {ep + 1:>3}/{epochs}  pred={tot['pred']:.5f} "
                  f"var={tot['var']:.4f} cov={tot['cov']:.4f}")
    return model.cpu(), hist


def fit_position_probe(model, seqs):
    """Frozen linear probe latent z -> position (m): the channel-chart readout.

    Least-squares fit on encoded latents; measures how metric the learned chart is
    and lets us score the imagined rollout in physical units. Trained on the SAME
    train walks (probe is a readout, not part of the world model)."""
    import torch

    model.eval()
    Z, P = [], []
    with torch.no_grad():
        for s in seqs:
            z = model.encode(torch.from_numpy(s["feat"])).numpy()
            Z.append(z)
            P.append(s["pos"])
    Z = np.concatenate(Z)
    P = np.concatenate(P)
    Zb = np.concatenate([Z, np.ones((Z.shape[0], 1), np.float32)], axis=1)
    W, *_ = np.linalg.lstsq(Zb, P, rcond=None)  # (d+1, 2)
    fit_err = np.linalg.norm(Zb @ W - P, axis=1).mean()
    return W, float(fit_err)


def latent_to_pos(z, W):
    zb = np.concatenate([z, np.ones((z.shape[0], 1), np.float32)], axis=1)
    return zb @ W


def multistep_latent_error(model, seqs, *, L):
    """Held-out MULTI-STEP next-LATENT error (the fair JEPA metric).

    Encode the seed CSI once, roll the predictor L-1 steps under the actions, and
    measure the mean L2 gap to the target-encoder latents of the true CSI. This is
    the regime where the velocity action matters; the 1-step error is trivially
    small for any model because consecutive latents are nearly identical."""
    import torch

    model.eval()
    errs = []
    with torch.no_grad():
        for s in seqs:
            f = s["feat"]
            v = s["vel"]
            T = len(f)
            for start in range(0, T - L + 1):
                z = model.encode(torch.from_numpy(f[start : start + 1]))
                step_err = []
                for k in range(L - 1):
                    a = torch.from_numpy(v[start + k : start + k + 1])
                    z = model.predictor(z, a)
                    zt = model.encode_target(torch.from_numpy(f[start + k + 1 : start + k + 2]))
                    step_err.append(float(((z - zt) ** 2).sum().sqrt()))
                errs.append(np.mean(step_err))
    return float(np.mean(errs))


def imagine_vs_baseline(model, seqs, W, *, seed_k=2, steps=8):
    """Roll latent forward under the action seq; read out position; vs const-vel.

    For each held-out walk: encode CSI at the seed frame, then imagine `steps`
    ahead applying ONLY the velocity actions (no more CSI). Read position from the
    frozen probe. Baseline = constant-velocity extrapolation from the seed."""
    import torch

    model.eval()
    wm_errs, cv_errs = [], []
    for s in seqs:
        T = len(s["pos"])
        if T < seed_k + steps + 1:
            continue
        x_seed = torch.from_numpy(s["feat"][seed_k])
        actions = torch.from_numpy(s["vel"][seed_k : seed_k + steps])
        z_traj = rollout_latent(model, x_seed, actions).numpy()  # (steps+1, d)
        pos_pred = latent_to_pos(z_traj, W)                       # (steps+1, 2)
        gt = s["pos"][seed_k : seed_k + steps + 1]
        wm_errs.append(np.linalg.norm(pos_pred - gt, axis=1))
        # constant-velocity baseline from the last observed step
        v0 = s["pos"][seed_k] - s["pos"][seed_k - 1]
        cv = s["pos"][seed_k] + np.arange(steps + 1)[:, None] * v0[None, :]
        cv_errs.append(np.linalg.norm(cv - gt, axis=1))
    wm = np.stack(wm_errs)
    cv = np.stack(cv_errs)  # both (n_walks, steps+1)
    return wm.mean(axis=0), cv.mean(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seqs", type=int, default=40)
    ap.add_argument("--n-steps", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--train-len", type=int, default=6,
                    help="sub-rollout length L for multi-step JEPA training")
    ap.add_argument("--rollout-steps", type=int, default=14,
                    help="imagination horizon; the world-model's edge over "
                    "constant-velocity grows with horizon (the paper's claim)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    room = RoomConfig()
    fdim = feature_dim(room)
    print("=" * 64)
    print("CSI WORLD MODEL THROUGH RETINA  (SYNTHETIC CSI — documented forward model)")
    print("=" * 64)
    print(f"room: {room.width}x{room.depth} m, {room.n_sub} subcarriers x "
          f"{room.n_rx} antennas, fc={room.fc/1e9:.2f} GHz")
    print(f"CSI feature dim = {fdim} (|H|, cos/sin phase per subcarrier-antenna)\n")

    # ---- DATA ----
    all_seqs = build_dataset(room, n_seqs=args.n_seqs, n_steps=args.n_steps,
                             seed=args.seed)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(all_seqs))
    n_test = max(2, len(all_seqs) // 5)
    test = [all_seqs[i] for i in order[:n_test]]
    train = [all_seqs[i] for i in order[n_test:]]
    print(f"dataset: {len(all_seqs)} CSI walks "
          f"(train={len(train)}, test={len(test)}), seq_len={args.n_steps}\n")

    # ---- STATE THROUGH RETINA (real types) ----
    import torch
    enc0 = build_jepa(CSIConfig(feat_dim=fdim, latent_dim=args.latent_dim))
    with torch.no_grad():
        z0 = enc0.encode(torch.from_numpy(train[0]["feat"][0:1])).numpy()[0]
    ws = csi_worldstate(
        src="csi_room", t=0.0, frame=0,
        scene_latent=z0, scene_model="csi-jepa/v0",
        subject_pos=train[0]["pos"][0], subject_vel=train[0]["vel"][0],
        subject_latent=z0,
    )
    blob = ws.to_json()
    print("retina.WorldState for one CSI timestep (real types, serialized):")
    print(f"  {ws!r}")
    print(f"  scene.vec: model={ws.scene.model} dim={ws.scene.dim}")
    print(f"  entity: id={ws.entities[0].id} type={ws.entities[0].type} "
          f"bbox={ws.entities[0].bbox} locus={ws.entities[0].locus} "
          f"attrs={list(ws.entities[0].attrs)}")
    print(f"  serialized {len(blob)} bytes:\n  {blob[:160]}…\n")

    Xw, Aw = subrollout_windows(train, L=args.train_len)
    print(f"training windows: {Xw.shape[0]} sub-rollouts of length L={args.train_len} "
          f"(multi-step JEPA)\n")

    # ---- TRAIN: action-conditioned JEPA ----
    print("--- TRAIN action-conditioned JEPA (CSI + velocity action) ---")
    cfg = CSIConfig(feat_dim=fdim, latent_dim=args.latent_dim)
    m_ac = build_jepa(cfg, action_conditioned=True)
    m_ac, _ = train_jepa(m_ac, Xw, Aw, epochs=args.epochs, lr=args.lr,
                         batch=args.batch, device=args.device, seed=args.seed)

    # ---- ABLATION: CSI-only (action ignored) ----
    print("\n--- TRAIN ablation: CSI-only predictor (velocity action IGNORED) ---")
    m_csi = build_jepa(cfg, action_conditioned=False)
    m_csi, _ = train_jepa(m_csi, Xw, Aw, epochs=args.epochs, lr=args.lr,
                         batch=args.batch, device=args.device, seed=args.seed)

    # ---- EVAL: multi-step next-latent prediction (held out) ----
    ac_lat = multistep_latent_error(m_ac, test, L=args.train_len)
    csi_lat = multistep_latent_error(m_csi, test, L=args.train_len)

    # ---- channel-chart probe + imagination rollout vs baseline ----
    W_ac, fit_ac = fit_position_probe(m_ac, train)
    wm_curve, cv_curve = imagine_vs_baseline(
        m_ac, test, W_ac, seed_k=2, steps=args.rollout_steps)

    print("\n" + "=" * 64)
    print("RESULTS (held-out CSI walks)")
    print("=" * 64)
    print(f"(1) MULTIMODAL FUSION ABLATION — {args.train_len-1}-step next-latent "
          "L2 error (lower=better):")
    print(f"    action-conditioned (CSI + velocity) : {ac_lat:.4f}")
    print(f"    CSI-only           (action ignored) : {csi_lat:.4f}")
    impr = (csi_lat - ac_lat) / csi_lat * 100
    print(f"    => fusing the velocity action improves next-latent by {impr:+.1f}%")
    verdict = "HELPS" if ac_lat < csi_lat else "did NOT help"
    print(f"    VERDICT: the velocity action {verdict}.\n")

    print(f"(2) CHANNEL CHART — linear probe latent->position fit error: "
          f"{fit_ac:.3f} m")
    print("    (low => the JEPA latent self-organized into a metric room map)\n")

    print(f"(3) IMAGINATION ROLLOUT — {args.rollout_steps} steps, latent rolled "
          "under actions, position read from probe:")
    print(f"    {'step':>4}  {'world-model(m)':>14}  {'const-vel(m)':>12}")
    for k in range(len(wm_curve)):
        print(f"    {k:>4}  {wm_curve[k]:>14.3f}  {cv_curve[k]:>12.3f}")
    wm_m = float(wm_curve.mean())
    cv_m = float(cv_curve.mean())
    print(f"    {'mean':>4}  {wm_m:>14.3f}  {cv_m:>12.3f}")
    if wm_m < cv_m:
        print(f"    VERDICT: world-model beats constant-velocity by "
              f"{(cv_m - wm_m)/cv_m*100:.1f}% over the rollout.")
    else:
        print("    VERDICT: did not beat constant-velocity on this run.")
    print("=" * 64)


if __name__ == "__main__":
    main()
