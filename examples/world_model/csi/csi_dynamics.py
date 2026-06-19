"""CSI dynamics = the shared action-conditioned JEPA, specialized to CSI.

The action-conditioned, latent-predicting JEPA world model is **not** CSI-specific,
so it now lives as a reusable example module one level up:
[`examples/world_model/latent_dynamics.py`](../latent_dynamics.py). This file is the
thin CSI adapter over it — it picks the CSI defaults (a velocity action `a_t`, a
small channel-chart latent) and re-exports the shared builder/losses/rollout so the
rest of the demo imports them from here.

Why a *separate* dynamics example at all (vs the shipped `dynamics_model.py`):

  dynamics_model.py (appearance ablation)   latent_dynamics.py (this demo uses it)
  ------------------------------------      ------------------------------------------
  p(s_{t+1} | s_t), NO action               p(z_{t+1} | z_t, a_t), ACTION-conditioned
  predicts a POSITION delta (dx,dy)         predicts the next LATENT z_{t+1} (JEPA)
  token-per-(entity,timestep) slot grid     a single field latent per timestep
  trained by MSE on positions               latent MSE + VICReg anti-collapse

Faithful to the two reference papers (see README): JEPA latent prediction with an
EMA target encoder + VICReg (arXiv:2409.10045), and an action-conditioned
homomorphic (Lie-algebra, `z_{t+1}=exp(A(a))z_t+b(a)`) transition (arXiv:2603.20048).

All torch stays lazily imported inside the shared module (the `[dynamics]` extra),
so importing this adapter is numpy-free.
"""

from __future__ import annotations

import os
import sys

# Make the sibling `examples/world_model/` importable when this demo is run from
# its own directory (the README's `cd examples/world_model/csi`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latent_dynamics import (  # noqa: E402  (path set up just above)
    LatentDynamicsConfig,
    build_jepa,
    jepa_loss,
    jepa_loss_multistep,
    rollout_latent,
    vicreg_terms,
)


def CSIConfig(  # noqa: N802 — kept callable-with-same-name for the demo's call sites
    feat_dim: int,
    latent_dim: int = 16,
    action_dim: int = 2,
    enc_hidden: int = 128,
) -> LatentDynamicsConfig:
    """CSI specialization of `LatentDynamicsConfig`.

    Defaults to a 2-D velocity action (vx, vy) and a small channel-chart latent —
    the CSI-flavoured choices. The returned config is the generic, signal-agnostic
    `LatentDynamicsConfig`; nothing below it knows it's CSI.
    """
    return LatentDynamicsConfig(
        feat_dim=feat_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
        enc_hidden=enc_hidden,
    )


__all__ = [
    "CSIConfig",
    "LatentDynamicsConfig",
    "build_jepa",
    "jepa_loss",
    "jepa_loss_multistep",
    "rollout_latent",
    "vicreg_terms",
]
