"""Synthetic WiFi CSI forward model + a tiny CSI→latent encoder.

WHY SYNTHETIC. The two reference papers train on real CSI (DICHASUS etc.), but a
self-contained, offline, CI-runnable demo can't download a multi-GB measurement
set. So we *synthesize* CSI from a DOCUMENTED physical forward model and label it
loudly as synthetic. The forward model is deliberately faithful enough that the
two physical regularities the papers exploit are actually present in the data:

  * **channel charting** — CSI varies smoothly with the subject's position, so a
    good latent space becomes a metric map of the room layout (paper 2603.20048).
  * **action = velocity** — the subject moves with a velocity `a_t`; that action
    is exactly what advances the channel from H_t to H_{t+1} (the action-
    conditioned transition the homomorphic-world-model paper conditions on).

FORWARD MODEL (multipath, narrowband-per-subcarrier).
A single Tx antenna and `n_rx` Rx antennas (a small ULA) observe a room. There is
one line-of-sight path to a moving point scatterer (the subject) plus a few fixed
static multipath reflectors (walls/furniture). For subcarrier f and Rx antenna r
the channel is the coherent sum of paths:

    H[f, r] = sum_p  a_p * exp(-j * 2*pi * f * tau_p) * exp(-j * 2*pi * d_r * sin(theta_p) / lambda)

  * `a_p`   path amplitude ~ 1 / dist_p   (free-space-ish attenuation)
  * `tau_p` propagation delay = dist_p / c (gives the per-subcarrier phase slope)
  * `theta_p` angle-of-arrival at the array (gives the per-antenna phase ramp)
  * the moving subject's path has `dist`/`theta` that change as it walks; static
    reflectors are constant — so H_t carries a position-dependent signature.

The subject walks a smooth trajectory (slowly-varying velocity = the action). We
return, per timestep: the complex CSI tensor H_t (n_sub x n_rx), the subject
position p_t, and the velocity a_t. Everything is numpy; fully offline.

This is NOT a calibrated channel simulator (no Doppler, no fading statistics, no
real antenna patterns). It is the *minimum* forward model that makes the two
physical regularities real, so the world model has something true to learn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C = 3e8  # speed of light, m/s


@dataclass
class RoomConfig:
    """Geometry + radio config for the synthetic room. SI units (metres, Hz)."""

    width: float = 6.0   # room x-extent (m)
    depth: float = 5.0   # room y-extent (m)
    n_sub: int = 32      # number of OFDM subcarriers
    n_rx: int = 8        # Rx antennas in the ULA (small 8-element array)
    fc: float = 5.18e9   # carrier (WiFi ch.36, 5 GHz)
    bw: float = 20e6     # occupied bandwidth (subcarrier spacing = bw/n_sub)
    array_dx: float = 0.029  # antenna spacing ~ lambda/2 at 5.18 GHz
    subject_gain: float = 6.0  # reflectivity of the moving subject's LOS path
    # Fixed static reflectors (x, y, reflectivity) — walls/furniture. Constant in
    # time, so they form the static "map" the channel chart factors out. Kept WEAK
    # relative to the moving subject: strong static multipath makes the CSI->room
    # map many-to-one (a diagnostic we actually hit — a stationary cluster of
    # bright reflectors swamps the subject's signature and position stops being
    # decodable). A LOS-dominant sensing geometry keeps the chart learnable, which
    # is the regime WiFi-sensing/channel-charting work targets.
    reflectors: list[tuple[float, float, float]] = field(
        default_factory=lambda: [
            (3.0, 5.0, 0.4),   # far wall
            (6.0, 2.5, 0.4),   # right wall
        ]
    )
    array_pos: tuple[float, float] = (3.0, 0.0)  # Rx array at near wall, centre

    @property
    def lambda_c(self) -> float:
        return C / self.fc

    @property
    def subcarrier_freqs(self) -> np.ndarray:
        # absolute subcarrier frequencies centred on fc
        offs = (np.arange(self.n_sub) - self.n_sub / 2) * (self.bw / self.n_sub)
        return self.fc + offs


def _csi_for_position(cfg: RoomConfig, p: np.ndarray, rng: np.random.Generator,
                      noise: float) -> np.ndarray:
    """Complex CSI tensor H (n_sub, n_rx) for the subject at position `p`.

    Coherent sum of the moving subject's LOS path and the fixed reflector paths,
    each contributing a per-subcarrier delay phase and a per-antenna AoA phase.
    """
    ax, ay = cfg.array_pos
    freqs = cfg.subcarrier_freqs                       # (n_sub,)
    ant = (np.arange(cfg.n_rx) - (cfg.n_rx - 1) / 2) * cfg.array_dx  # (n_rx,)

    H = np.zeros((cfg.n_sub, cfg.n_rx), dtype=np.complex128)

    # paths: (x, y, amplitude-scale). The subject is the strong moving path;
    # the reflectors are weak fixed paths (see RoomConfig.reflectors note).
    paths = [(p[0], p[1], cfg.subject_gain)]
    paths += [(rx, ry, refl) for (rx, ry, refl) in cfg.reflectors]

    for (px, py, scale) in paths:
        dx, dy = px - ax, py - ay
        dist = float(np.hypot(dx, dy)) + 1e-3
        theta = np.arctan2(dx, dy)             # AoA from array boresight (+y)
        amp = scale / dist                     # ~ free-space attenuation
        tau = dist / C                          # propagation delay
        # per-subcarrier delay phase  exp(-j 2pi f tau)
        delay_phase = np.exp(-1j * 2 * np.pi * freqs * tau)          # (n_sub,)
        # per-antenna steering phase  exp(-j 2pi d sin(theta) / lambda)
        steer = np.exp(-1j * 2 * np.pi * ant * np.sin(theta) / cfg.lambda_c)  # (n_rx,)
        H += amp * np.outer(delay_phase, steer)

    if noise > 0:
        H = H + noise * (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape))
    return H


def _smooth_walk(cfg: RoomConfig, n_steps: int, rng: np.random.Generator,
                 speed: float) -> tuple[np.ndarray, np.ndarray]:
    """A smooth random walk inside the room. Returns (pos (T,2), vel (T,2)).

    Velocity is the ACTION: a low-pass random walk so motion is smooth and the
    next position is well predicted by the current one plus the action. Reflects
    off the walls (a margin) so the subject stays inside the room.
    """
    margin = 0.4
    pos = np.zeros((n_steps, 2), np.float64)
    vel = np.zeros((n_steps, 2), np.float64)
    p = np.array([rng.uniform(margin, cfg.width - margin),
                  rng.uniform(margin, cfg.depth - margin)])
    v = rng.standard_normal(2)
    v = v / (np.linalg.norm(v) + 1e-9) * speed
    for t in range(n_steps):
        # low-pass the heading so the walk curves smoothly (momentum + jitter)
        v = 0.85 * v + 0.15 * rng.standard_normal(2) * speed
        nrm = np.linalg.norm(v)
        if nrm > 1e-9:
            v = v / nrm * speed
        np_ = p + v
        # reflect off room bounds
        for d, lim in ((0, cfg.width), (1, cfg.depth)):
            if np_[d] < margin:
                np_[d] = margin + (margin - np_[d])
                v[d] = -v[d]
            elif np_[d] > lim - margin:
                np_[d] = (lim - margin) - (np_[d] - (lim - margin))
                v[d] = -v[d]
        pos[t] = p
        vel[t] = np_ - p   # the action that takes p_t -> p_{t+1}
        p = np_
    return pos, vel


def make_csi_sequence(cfg: RoomConfig, *, n_steps: int, seed: int,
                      speed: float = 0.18, noise: float = 0.004) -> dict:
    """One walk: per-timestep complex CSI, subject position, and velocity action.

    Returns a dict with:
      H    (T, n_sub, n_rx) complex64  — the CSI tensors (SYNTHETIC)
      pos  (T, 2) float32              — ground-truth subject position (m)
      vel  (T, 2) float32              — velocity action a_t (m/step)
    """
    rng = np.random.default_rng(seed)
    pos, vel = _smooth_walk(cfg, n_steps, rng, speed)
    H = np.zeros((n_steps, cfg.n_sub, cfg.n_rx), np.complex64)
    for t in range(n_steps):
        H[t] = _csi_for_position(cfg, pos[t], rng, noise).astype(np.complex64)
    return {
        "H": H,
        "pos": pos.astype(np.float32),
        "vel": vel.astype(np.float32),
    }


def csi_to_features(H: np.ndarray) -> np.ndarray:
    """Flatten a complex CSI tensor (n_sub, n_rx) to a real feature vector.

    Real measurement pipelines feed amplitude + (sanitised) phase. We mirror that:
    per (subcarrier, antenna) we emit |H| and the cos/sin of its phase — a
    continuous, wrap-free phase encoding. Output length = n_sub*n_rx*3.
    """
    amp = np.abs(H)
    ph = np.angle(H)
    feat = np.stack([amp, np.cos(ph), np.sin(ph)], axis=-1)  # (n_sub, n_rx, 3)
    return feat.reshape(-1).astype(np.float32)


def feature_dim(cfg: RoomConfig) -> int:
    return cfg.n_sub * cfg.n_rx * 3
