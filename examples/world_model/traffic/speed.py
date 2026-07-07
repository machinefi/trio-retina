"""Speed from Retina state — differentiate `Entity.locus` over time.

Once every vehicle carries a metric ground position on `Entity.locus` (metres,
see `homography.py`), speed is just calculus: track each id's locus through the
`WorldState` stream and take a smoothed time-derivative. `|d locus / dt|` is
metres per second; ×3.6 is km/h.

This is deliberately an **example**, not core: `speed` is a domain verb built
*from* Retina primitives (tracked entities + a `line.cross`-style trap), keeping
the library's app-agnostic boundary intact. The estimator emits a `retina.event`
`Event(type="speed", ...)` the moment a vehicle crosses the measurement line —
the exact analog of a roadside radar's trigger, but computed from state.

Pure numpy — runs in the offline core with no camera, no model.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from retina import Event


def _finite_diff_speed(times: list[float], loci: list[tuple[float, ...]]) -> float:
    """Least-squares speed (m/s) over a short window of (t, locus) samples.

    Fits X(t) and Y(t) linearly and returns the ground-plane speed
    √(vx² + vy²). Least squares over a few frames is far steadier than a raw
    two-point difference, which jitters with per-frame detection noise.
    """
    if len(times) < 2:
        return 0.0
    t = np.asarray(times, np.float64)
    t = t - t[0]
    p = np.asarray(loci, np.float64)
    # slope of a 1st-degree fit per axis = velocity component
    vx = np.polyfit(t, p[:, 0], 1)[0]
    vy = np.polyfit(t, p[:, 1], 1)[0]
    return float(np.hypot(vx, vy))


@dataclass
class SpeedEstimator:
    """Streaming per-vehicle speed from a sequence of `WorldState`s.

    Feed it `(t, world_state)` in order. It maintains a short locus history per
    entity id, annotates each entity in place with `attrs["speed_kmh"]`, and
    fires a `speed` `Event` when a vehicle's ground X crosses `trap_x` (metres) —
    a virtual speed trap. `window` is how many recent samples the fit uses.

    `src` labels the event source (the camera id). `min_samples` gates noisy
    first frames; a vehicle needs that many locus samples before it gets a speed.
    """

    src: str = "cam"
    trap_x: float | None = None
    window: int = 6
    min_samples: int = 3
    # Metric region of interest (xmin, xmax, ymin, ymax) in the same units as
    # locus. A planar homography *diverges near the horizon*: detections above
    # the calibrated road patch project to absurd metre coordinates (and absurd
    # speeds). Gating on the calibrated ROI drops exactly those, which is the
    # honest thing to do — you can only measure where you calibrated.
    roi_m: tuple[float, float, float, float] | None = None
    _t_hist: dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=32)))
    _p_hist: dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=32)))
    _last_x: dict[str, float] = field(default_factory=dict)
    _tripped: set[str] = field(default_factory=set)
    _pending: set[str] = field(default_factory=set)

    def kmh(self, eid: str, min_samples: int | None = None) -> float | None:
        """Current smoothed speed for an id, or None if not enough samples.

        `min_samples` overrides the instance default — the trap uses a lower bar
        (2) so a car that crosses `trap_x` early still gets a reading rather than
        being dropped."""
        ts = list(self._t_hist[eid])[-self.window :]
        ps = list(self._p_hist[eid])[-self.window :]
        need = self.min_samples if min_samples is None else min_samples
        if len(ts) < max(2, need):
            return None
        return _finite_diff_speed(ts, ps) * 3.6

    def update(self, t: float, ws) -> list[Event]:
        """Ingest one timestep; annotate entities; return any `speed` events."""
        events: list[Event] = []
        for ent in ws.entities:
            if ent.locus is None:
                continue
            if self.roi_m is not None:
                x0, y0 = float(ent.locus[0]), float(ent.locus[1])
                xmin, xmax, ymin, ymax = self.roi_m
                if not (xmin <= x0 <= xmax and ymin <= y0 <= ymax):
                    continue  # outside the calibrated patch — homography unreliable
            eid = ent.id
            self._t_hist[eid].append(float(t))
            self._p_hist[eid].append(tuple(ent.locus))
            v = self.kmh(eid)
            if v is not None:
                ent.attrs["speed_kmh"] = round(v, 1)

            # Virtual speed trap: fire once, when ground X crosses trap_x. The
            # crossing is latched in `_pending` so that if the sign-change lands
            # before enough samples exist for a speed, we still fire on the next
            # frame instead of losing the measurement.
            if self.trap_x is not None and eid not in self._tripped:
                x = float(ent.locus[0])
                prev = self._last_x.get(eid)
                crossed = prev is not None and (prev - self.trap_x) * (x - self.trap_x) <= 0
                if crossed or eid in self._pending:
                    tv = self.kmh(eid, min_samples=2)
                    if tv is not None:
                        self._tripped.add(eid)
                        self._pending.discard(eid)
                        events.append(
                            Event(
                                type="speed",
                                t=float(t),
                                src=self.src,
                                id=_as_int(eid),
                                label=ent.type,
                                box=ent.bbox,
                                ext={"kmh": round(tv, 1),
                                     "locus_m": [round(c, 2) for c in ent.locus]},
                            )
                        )
                    else:
                        self._pending.add(eid)  # retry next frame once speed exists
                self._last_x[eid] = x
        return events


def _as_int(eid: str) -> int | None:
    try:
        return int(eid)
    except (TypeError, ValueError):
        return None


def estimate_speeds(states, *, src="cam", trap_x=None, window=6):
    """Convenience: run a `SpeedEstimator` over a full `[(t, ws), ...]` list.

    Annotates every `ws` in place with per-entity `attrs["speed_kmh"]` and
    returns the flat list of `speed` events collected across the sequence.
    """
    est = SpeedEstimator(src=src, trap_x=trap_x, window=window)
    events: list[Event] = []
    for t, ws in states:
        events.extend(est.update(t, ws))
    return events
