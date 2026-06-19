"""Smoke tests for the CSI world-model demo — keep it actually runnable.

Run from this directory so the sibling modules import:

    cd examples/world_model/csi && pytest test_csi_demo.py

torch-dependent tests skip cleanly if the [dynamics] extra isn't installed, so
the numpy-only parts (forward model + Retina state assembly) always run.
"""

from __future__ import annotations

import numpy as np
import pytest

from csi_data import (
    RoomConfig,
    csi_to_features,
    feature_dim,
    make_csi_sequence,
)
from csi_state import csi_worldstate


def test_forward_model_shapes():
    cfg = RoomConfig()
    seq = make_csi_sequence(cfg, n_steps=20, seed=0)
    assert seq["H"].shape == (20, cfg.n_sub, cfg.n_rx)
    assert seq["H"].dtype == np.complex64
    assert seq["pos"].shape == (20, 2)
    assert seq["vel"].shape == (20, 2)
    # subject stays inside the room
    assert seq["pos"][:, 0].min() >= 0 and seq["pos"][:, 0].max() <= cfg.width
    assert seq["pos"][:, 1].min() >= 0 and seq["pos"][:, 1].max() <= cfg.depth


def test_features_real_and_sized():
    cfg = RoomConfig()
    seq = make_csi_sequence(cfg, n_steps=3, seed=1)
    f = csi_to_features(seq["H"][0])
    assert f.shape == (feature_dim(cfg),)
    assert f.dtype == np.float32
    assert np.isfinite(f).all()


def test_csi_carries_position_signal():
    """Two DIFFERENT positions must give different CSI; the action advances it.

    A weak guarantee that the forward model is position-informative (not constant),
    so a channel chart is learnable at all."""
    cfg = RoomConfig()
    seq = make_csi_sequence(cfg, n_steps=30, seed=2)
    f0 = csi_to_features(seq["H"][0])
    fmid = csi_to_features(seq["H"][15])
    assert np.linalg.norm(f0 - fmid) > 1e-3


def test_worldstate_uses_real_retina_types_and_roundtrips():
    """The CSI state rides Retina's real Vec/Entity/WorldState and serializes."""
    from retina import Vec, WorldState

    cfg = RoomConfig()
    seq = make_csi_sequence(cfg, n_steps=2, seed=3)
    z = np.arange(8, dtype=np.float32) / 8
    ws = csi_worldstate(
        src="csi_room", t=1.0, frame=0,
        scene_latent=z, scene_model="csi-jepa/test",
        subject_pos=seq["pos"][0], subject_vel=seq["vel"][0],
        subject_latent=z,
    )
    assert isinstance(ws, WorldState)
    assert isinstance(ws.scene, Vec)
    assert ws.scene.dim == 8
    e = ws.entities[0]
    assert e.bbox is None  # CSI has no pixel bbox — it's a field source
    # NATIVE: metric position rides the typed `locus`, not the attrs bag.
    assert e.locus is not None and len(e.locus) == 2
    assert "pos_m" not in e.attrs  # no longer attr-stuffed
    assert "vel_action_m" in e.attrs  # action stays in attrs (transition input)
    d = ws.to_dict()
    assert d["scene"]["model"] == "csi-jepa/test"
    assert d["entities"][0]["locus"] == list(e.locus)
    assert d["entities"][0]["vec"]["dim"] == 8


def test_jepa_trains_one_step():
    torch = pytest.importorskip("torch")
    from csi_dynamics import CSIConfig, build_jepa, jepa_loss_multistep

    cfg = RoomConfig()
    fdim = feature_dim(cfg)
    seq = make_csi_sequence(cfg, n_steps=8, seed=4)
    feats = np.stack([csi_to_features(h) for h in seq["H"]]).astype(np.float32)
    x = torch.from_numpy(feats[None, :6])          # (1, L=6, feat)
    a = torch.from_numpy(seq["vel"][None, :5])     # (1, L-1, 2)
    model = build_jepa(CSIConfig(feat_dim=fdim, latent_dim=8), action_conditioned=True)
    loss, parts = jepa_loss_multistep(model, x, a)
    loss.backward()
    assert np.isfinite(parts["pred"])
    assert all(p.grad is not None for p in model.online.parameters())


def test_homomorphic_predictor_is_norm_stable():
    """The skew-symmetric (orthogonal) transition must not blow up over a long
    rollout — the failure mode an unconstrained generator hits."""
    torch = pytest.importorskip("torch")
    from csi_dynamics import CSIConfig, build_jepa

    model = build_jepa(CSIConfig(feat_dim=10, latent_dim=8), action_conditioned=True)
    z = torch.randn(1, 8)
    a = torch.randn(1, 2)
    n0 = z.norm().item()
    with torch.no_grad():
        for _ in range(50):
            z = model.predictor(z, a)
    # bias term can shift it, but it must stay finite and bounded (not exploding)
    assert torch.isfinite(z).all()
    assert z.norm().item() < n0 * 50 + 100
