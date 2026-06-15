"""The dynamics seam — predict the next state from a stream of WorldStates.

This is the L4 layer that sits *on top of* Retina (not in the core). The point of
the demo: a dynamics model consumes Retina's structured `WorldState` (per-entity),
NOT raw pixels — so Retina is the necessary, model-agnostic interface between any
backbone and any dynamics engine. The `DynamicsModel` protocol keeps the dynamics
swappable too: a pure-Python baseline now, TD-MPC2 / an object-centric model later.
"""

from __future__ import annotations

from typing import Protocol

from retina import Entity, WorldState


class DynamicsModel(Protocol):
    """Observe a stream of states, then predict the state `horizon_s` ahead."""

    def observe(self, state: WorldState) -> None: ...
    def predict(self, horizon_s: float) -> WorldState: ...


class LinearForecaster:
    """Per-entity constant-velocity baseline — the simplest dynamics on a
    `WorldState`. Extrapolates each entity's centroid by its recent velocity. It
    is the bar a real engine (TD-MPC2, object-centric) has to beat; it also proves
    the harness end-to-end with zero deps."""

    def __init__(self, history: int = 5):
        self._history = history
        self._hist: dict[str, list[tuple]] = {}  # id -> [(t, cx, cy, w, h)]
        self._type: dict[str, str] = {}
        self._last_t = 0.0

    def observe(self, state: WorldState) -> None:
        self._last_t = state.t
        for e in state.entities:
            if e.bbox is None:
                continue
            x1, y1, x2, y2 = e.bbox
            track = self._hist.setdefault(e.id, [])
            track.append((state.t, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
            del track[: -self._history]
            self._type[e.id] = e.type

    def predict(self, horizon_s: float) -> WorldState:
        entities = []
        for eid, h in self._hist.items():
            if len(h) < 2:
                continue
            (t0, cx0, cy0, _, _), (t1, cx1, cy1, w, hh) = h[-2], h[-1]
            dt = (t1 - t0) or 1e-9
            vx, vy = (cx1 - cx0) / dt, (cy1 - cy0) / dt
            cx, cy = cx1 + vx * horizon_s, cy1 + vy * horizon_s
            entities.append(
                Entity(id=eid, type=self._type.get(eid, "?"),
                       bbox=(cx - w / 2, cy - hh / 2, cx + w / 2, cy + hh / 2))
            )
        return WorldState(src="forecast", t=self._last_t + horizon_s, entities=entities)


class LearnedForecaster:
    """The TRAINED dynamics (the MLP from train_dynamics.py) — anticipates turns /
    slowdowns from the recent track window, not just constant velocity. Loads the
    saved weights; needs torch (the render venv has it). Same `DynamicsModel` seam."""

    def __init__(self, weights: str):
        import torch
        import torch.nn as nn

        ck = torch.load(weights, map_location="cpu", weights_only=False)
        self.W, self.wh = ck["W"], ck["wh"]
        net = nn.Sequential(
            nn.Linear(self.W * 2, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2)
        )
        net.load_state_dict({k[len("net."):]: v for k, v in ck["sd"].items()})
        net.eval()
        self._torch, self._net = torch, net
        self._hist: dict[str, list] = {}
        self._type: dict[str, str] = {}

    def observe(self, state: WorldState) -> None:
        for e in state.entities:
            if e.bbox is None:
                continue
            x1, y1, x2, y2 = e.bbox
            h = self._hist.setdefault(e.id, [])
            h.append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
            del h[: -self.W]
            self._type[e.id] = e.type

    def predict(self, horizon_s: float = 0.0) -> WorldState:
        import numpy as np

        entities = []
        for eid, h in self._hist.items():
            if len(h) < self.W:
                continue
            p = np.array([[c[0] / self.wh[0], c[1] / self.wh[1]] for c in h[-self.W:]], dtype="float32")
            with self._torch.no_grad():
                x = self._torch.tensor((p - p[-1]).reshape(1, -1))  # (1, W*2)
                d = self._net(x).squeeze(0).numpy()
            cx, cy, w, hh = h[-1]
            ncx, ncy = cx + float(d[0]) * self.wh[0], cy + float(d[1]) * self.wh[1]
            entities.append(Entity(id=eid, type=self._type.get(eid, "?"),
                                   bbox=(ncx - w / 2, ncy - hh / 2, ncx + w / 2, ncy + hh / 2)))
        return WorldState(src="forecast", t=0.0, entities=entities)


def centroid(e: Entity) -> tuple[float, float] | None:
    if e.bbox is None:
        return None
    x1, y1, x2, y2 = e.bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def forecast_error(pred: WorldState, actual: WorldState) -> dict:
    """Mean centroid error (pixels) over entities present in BOTH states, matched
    by id. Returns {mae, matched, missed}."""
    actual_by_id = {e.id: e for e in actual.entities}
    errors, missed = [], 0
    for p in pred.entities:
        a = actual_by_id.get(p.id)
        cp, ca = centroid(p), centroid(a) if a else None
        if cp is None or ca is None:
            missed += 1
            continue
        errors.append(((cp[0] - ca[0]) ** 2 + (cp[1] - ca[1]) ** 2) ** 0.5)
    return {
        "mae": round(sum(errors) / len(errors), 2) if errors else None,
        "matched": len(errors),
        "missed": missed,
    }


class TDMPC2Dynamics:
    """Adapter to plug TD-MPC2's world model in behind the same `DynamicsModel`
    seam. TD-MPC2 ingests a structured state vector (exactly what Retina produces)
    and has no real-world perception front-end — which is why Retina is the
    necessary interface. Heavy; run on the Mac Studio (torch MPS).

        pip install tdmpc2   # + torch with MPS

    Implementation note (for the GPU/MPS run): featurize each WorldState into a
    fixed-width per-entity vector [cx, cy, w, h, (+ optional vec)], pad/pool to the
    model's obs dim, and roll the latent dynamics forward `horizon` steps, then map
    the predicted latent back to per-entity positions. Left as the Mac-Studio task;
    the protocol + featurizer below are the stable seam.
    """

    def __init__(self, *args, **kwargs):  # pragma: no cover - needs torch + tdmpc2
        raise NotImplementedError(
            "TDMPC2Dynamics runs on the Mac Studio (torch MPS + tdmpc2). "
            "Use LinearForecaster for the offline/local harness."
        )
