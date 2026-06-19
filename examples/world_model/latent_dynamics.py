"""Action-conditioned JEPA latent dynamics — a reusable, signal-agnostic variant.

This is the *second* dynamics example, beside `dynamics_model.py`. Where that one
is the focused appearance-ablation transformer (`p(s_{t+1} | s_t)`, no action,
predicts a position delta), this one is the **action-conditioned, latent-predicting
JEPA** world model that controlled / field signals need:

    dynamics_model.py                  latent_dynamics.py (this file)
    -------------------------------    --------------------------------------
    p(s_{t+1} | s_t), NO action        p(z_{t+1} | z_t, a_t), ACTION-conditioned
    predicts a POSITION delta (dx,dy)  predicts the next LATENT z_{t+1} (JEPA)
    token-per-(entity,timestep)        a single field latent per timestep (no slots)
    trained by MSE on positions        trained by latent MSE + VICReg anti-collapse

It is **signal-agnostic**: it takes a `latent_dim`, an `action_dim`, and an input
`feat_dim`, and knows nothing about CSI, vision, or any modality. The CSI demo
([`csi/`](csi/)) imports it to carry a WiFi channel latent; the same module would
carry a V-JEPA scene latent under a control action with no change.

Recipe (faithful to two CSI world-model papers, see `csi/README.md`):
  * **JEPA** (arXiv:2409.10045): an encoder maps the signal -> latent `z`, a
    predictor maps `z_t -> z_{t+1}` trained purely in LATENT space (no raw signal
    reconstruction); the latent self-organizes into a chart. Anti-collapse via an
    EMA target encoder + VICReg (variance + covariance regularization).
  * **Homomorphic transition** (arXiv:2603.20048): the predictor is conditioned on
    the action `a_t` as a Lie-algebra update `z_{t+1} = exp(A(a_t)) @ z_t + b(a_t)`.
    `torch.matrix_exp` of a skew-symmetric generator is ORTHOGONAL, so the update is
    norm-preserving and cannot blow up over a long imagination rollout.

All torch is imported lazily (the `[dynamics]` extra), so `import retina` and even
importing this module stay numpy-free until a builder/loss is actually called.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LatentDynamicsConfig:
    """Shape of a latent-dynamics world model. Signal-agnostic.

    `feat_dim`   — raw input feature length the encoder consumes.
    `latent_dim` — the JEPA latent `z` dimension (the chart).
    `action_dim` — the control/action dimension `a_t` the transition conditions on.
    `enc_hidden` — encoder/predictor MLP hidden width.
    """

    feat_dim: int
    latent_dim: int = 16
    action_dim: int = 2
    enc_hidden: int = 128


def build_jepa(cfg: LatentDynamicsConfig, *, action_conditioned: bool = True):
    """Construct the JEPA: (online encoder, EMA target encoder, predictor) as one Module.

    `action_conditioned=True` makes the predictor a homomorphic action update
    `z_{t+1} = exp(A(a_t)) @ z_t + b(a_t)`. `False` is the ABLATION: the predictor
    ignores the action and is a plain residual MLP `z_t -> z_{t+1}`, to show the
    action channel actually helps next-latent prediction.
    """
    import torch
    import torch.nn as nn

    class Encoder(nn.Module):
        """Signal features -> latent z. Small MLP on purpose."""

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(cfg.feat_dim, cfg.enc_hidden),
                nn.GELU(),
                nn.Linear(cfg.enc_hidden, cfg.enc_hidden),
                nn.GELU(),
                nn.Linear(cfg.enc_hidden, cfg.latent_dim),
            )

        def forward(self, x):
            return self.net(x)

    class HomomorphicPredictor(nn.Module):
        """Action-conditioned transition z_{t+1} = exp(A(a)) @ z_t + b(a).

        The action `a_t` is mapped to a generator A(a) in the Lie algebra;
        `torch.matrix_exp` lifts it to a group element exp(A(a)) acting on the
        latent. A small action-driven bias b(a) lets the action *translate* the
        chart, while exp(A) gives the structured rotation/scaling part. exp of a sum
        composes, so chained actions compose (compositionality).

        Stability: A is built *skew-symmetric*, A = S - S^T. exp of a skew-symmetric
        matrix is ORTHOGONAL (a pure rotation), so ||exp(A) z|| = ||z|| exactly — the
        transition can never blow the latent up over a long imagination rollout.
        """

        def __init__(self):
            super().__init__()
            d = cfg.latent_dim
            self.gen = nn.Sequential(
                nn.Linear(cfg.action_dim, cfg.enc_hidden),
                nn.GELU(),
                nn.Linear(cfg.enc_hidden, d * d),
            )
            self.bias = nn.Sequential(
                nn.Linear(cfg.action_dim, cfg.enc_hidden),
                nn.GELU(),
                nn.Linear(cfg.enc_hidden, d),
            )
            # start near identity (zero generator, zero translation) for stability
            nn.init.zeros_(self.gen[-1].weight)
            nn.init.zeros_(self.gen[-1].bias)
            nn.init.zeros_(self.bias[-1].weight)
            nn.init.zeros_(self.bias[-1].bias)
            self.d = d

        def forward(self, z, a):
            b = z.shape[0]
            S = self.gen(a).reshape(b, self.d, self.d)
            A = S - S.transpose(1, 2)               # skew-symmetric generator
            G = torch.matrix_exp(A)                 # orthogonal group element
            rot = torch.bmm(G, z.unsqueeze(-1)).squeeze(-1)
            return rot + self.bias(a)               # rotate + action-driven shift

    class MLPPredictor(nn.Module):
        """ABLATION: predict z_{t+1} from z_t alone, action ignored."""

        def __init__(self):
            super().__init__()
            d = cfg.latent_dim
            self.net = nn.Sequential(
                nn.Linear(d, cfg.enc_hidden), nn.GELU(),
                nn.Linear(cfg.enc_hidden, d),
            )

        def forward(self, z, a):  # a accepted but ignored (same call signature)
            return z + self.net(z)   # residual, so identity is the easy default

    class JEPA(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_conditioned = action_conditioned
            self.online = Encoder()
            # target encoder: EMA copy, no grad (JEPA target tower)
            self.target = Encoder()
            self.target.load_state_dict(self.online.state_dict())
            for p in self.target.parameters():
                p.requires_grad_(False)
            self.predictor = HomomorphicPredictor() if action_conditioned else MLPPredictor()

        @torch.no_grad()
        def update_target(self, tau: float = 0.99):
            for tp, op in zip(self.target.parameters(), self.online.parameters(),
                              strict=True):
                tp.mul_(tau).add_(op, alpha=1 - tau)

        def encode(self, x):
            return self.online(x)

        @torch.no_grad()
        def encode_target(self, x):
            return self.target(x)

        def forward(self, x_t, a_t):
            z_t = self.online(x_t)
            z_pred = self.predictor(z_t, a_t)
            return z_t, z_pred

    return JEPA()


# --------------------------------------------------------------------------
# Losses: latent prediction + VICReg variance/covariance anti-collapse.
# --------------------------------------------------------------------------


def vicreg_terms(z):
    """Variance + covariance regularizers (Bardes et al. VICReg) to stop collapse.

    variance: hinge so each latent dim keeps std >= 1 (spread the chart).
    covariance: push off-diagonal feature covariances to 0 (decorrelate dims).
    """
    import torch

    if z.shape[0] < 2:  # variance undefined for a single sample
        zero = z.sum() * 0.0
        return zero, zero
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = torch.relu(1.0 - std).mean()
    zc = z - z.mean(dim=0, keepdim=True)
    n, d = zc.shape
    cov = (zc.T @ zc) / (n - 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = (off ** 2).sum() / d
    return var_loss, cov_loss


def jepa_loss_multistep(model, x_seq, a_seq, *, lam_var=1.0, lam_cov=0.04):
    """Multi-step JEPA loss over a short sub-rollout — where the ACTION earns it.

    `x_seq` (B, L, feat), `a_seq` (B, L-1, action). Encode x_0 ONCE, then roll the
    predictor forward L-1 steps applying the actions, and at every step match the
    rolled latent to the TARGET-encoder latent of the true signal at that step. An
    action-blind predictor cannot integrate where the subject goes over L steps; an
    action-conditioned one can. This is the regime that separates the two — the
    1-step task is trivially solvable by "predict z_t".

    No raw reconstruction: all supervision is in latent space (the paper's point).
    """
    import torch
    import torch.nn.functional as F

    b, L, _ = x_seq.shape
    z = model.online(x_seq[:, 0])
    var_l, cov_l = vicreg_terms(z)  # anti-collapse on the encoded seed latents
    pred_loss = 0.0
    with torch.no_grad():
        z_targets = [model.encode_target(x_seq[:, s]) for s in range(1, L)]
    for s in range(L - 1):
        z = model.predictor(z, a_seq[:, s])
        pred_loss = pred_loss + F.mse_loss(z, z_targets[s])
    pred_loss = pred_loss / (L - 1)
    loss = pred_loss + lam_var * var_l + lam_cov * cov_l
    return loss, {
        "pred": float(pred_loss.detach()),
        "var": float(var_l.detach()),
        "cov": float(cov_l.detach()),
    }


def jepa_loss(model, x_t, a_t, x_tp1, *, lam_var=1.0, lam_cov=0.04):
    """Single-step JEPA loss: predict the TARGET-encoder latent of x_{t+1} from (z_t, a_t).

    No raw signal reconstruction anywhere — supervision is entirely in latent space
    (the paper's point). Stop-gradient through the target tower.
    """
    import torch
    import torch.nn.functional as F

    z_t, z_pred = model(x_t, a_t)
    with torch.no_grad():
        z_tp1 = model.encode_target(x_tp1)
    pred_loss = F.mse_loss(z_pred, z_tp1)
    var_l, cov_l = vicreg_terms(z_t)
    loss = pred_loss + lam_var * var_l + lam_cov * cov_l
    return loss, {
        "pred": float(pred_loss.detach()),
        "var": float(var_l.detach()),
        "cov": float(cov_l.detach()),
    }


# --------------------------------------------------------------------------
# Imagination rollout in LATENT space (analogous to dynamics_model.rollout).
# --------------------------------------------------------------------------


def rollout_latent(model, x_seed, actions):
    """Imagine the latent forward from a seed signal, applying the action sequence.

    Encode the seed signal once to z_0, then autoregressively apply the (action-
    conditioned) predictor over `actions` (T, action_dim) WITHOUT ever seeing the
    signal again — pure imagination in latent space. Returns z_traj (T+1, latent_dim).

    This is the JEPA analogue of `dynamics_model.rollout()`: there it rolled raw
    positions; here it rolls the latent chart forward under actions.
    """
    import torch

    model.eval()
    with torch.no_grad():
        z = model.encode(x_seed.unsqueeze(0))  # (1, d)
        traj = [z.squeeze(0).clone()]
        for a in actions:
            z = model.predictor(z, a.unsqueeze(0))
            traj.append(z.squeeze(0).clone())
    return torch.stack(traj)
