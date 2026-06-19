"""Assemble a Retina `WorldState` from a CSI timestep — using the REAL types.

This is the "force CSI through Retina's abstractions" step. We take one synthetic
CSI timestep (the global channel latent + the moving subject) and express it as a
`retina.WorldState`, then serialize with the real `to_dict()`. Doing this honestly
surfaces exactly where the entity/bbox state model fits CSI and where it fights.

WHAT MAPS CLEANLY (validates Retina's design):
  * The **scene `Vec`** is a natural home for the GLOBAL channel latent — CSI is a
    whole-room field measurement, and `ws.scene` is precisely "a scene-level latent
    with no box". This is the single best fit in the whole exercise.
  * The dual **symbolic + latent** split holds: we keep a readable symbolic core
    (subject position, velocity) *and* the model-tagged `vec`, never collapsed.
  * `Vec` model-tagging (`model`, `dim`, `dtype`, `ref`) carries an RF latent with
    no schema change — a "csi-jepa" model tag sits beside "dinov2" without friction.

WHAT NOW MAPS NATIVELY (the framework closed the gap):
  * The subject's **metric position** (metres, room frame) rides the native
    `Entity.locus` — a typed world-frame coordinate distinct from the pixel `bbox`.
    A field signal no longer has to abuse `attrs` (or `bbox`) to express position.

WHERE IT STILL FIGHTS (honest remaining gap):
  * The ACTION (velocity) has nowhere first-class to live — `WorldState` is a pure
    state snapshot, and an action is a *transition input*, not state, so it does not
    belong on `WorldState`. We park it in `entity.attrs["vel_action_m"]` purely so
    the serialized frame is self-describing; the dynamics consumes the action
    out-of-band (see `csi_dynamics.py`). This is by design, not a schema gap.
"""

from __future__ import annotations

import numpy as np

from retina import Entity, Vec, WorldState


def csi_worldstate(
    *,
    src: str,
    t: float,
    frame: int,
    scene_latent: np.ndarray,
    scene_model: str,
    subject_pos: np.ndarray,
    subject_vel: np.ndarray,
    subject_latent: np.ndarray | None = None,
) -> WorldState:
    """Build a `WorldState` for one CSI timestep with the real Retina types.

    `scene_latent` is the global channel latent → `ws.scene` (the clean fit).
    The moving subject is an `Entity`; its metric position rides the native
    `Entity.locus`, and its CSI-derived per-object latent (if any) rides on
    `entity.vec`. The velocity action — a transition input, not state — is the only
    thing left in `attrs`.
    """
    scene = Vec(
        model=scene_model,
        dim=int(scene_latent.shape[0]),
        values=[round(float(v), 5) for v in scene_latent],
    )

    subj_vec = None
    if subject_latent is not None:
        subj_vec = Vec(
            model=scene_model + "/subj",
            dim=int(subject_latent.shape[0]),
            values=[round(float(v), 5) for v in subject_latent],
        )

    subject = Entity(
        id="subject",
        type="rf_subject",          # a field source, not a vision class
        bbox=None,                  # no pixel bbox — CSI is not a pixel detection
        # NATIVE: the subject's metric position (metres, room frame) is a first-class
        # typed `locus`, distinct from the pixel `bbox` — no attrs abuse.
        locus=tuple(round(float(v), 4) for v in subject_pos),
        conf=None,
        attrs={
            # The ACTION (velocity) is a transition input, not state, so it does not
            # belong on WorldState; we carry it here only so the serialized frame is
            # self-describing. Position is no longer here — it rides `locus`.
            "vel_action_m": [round(float(v), 4) for v in subject_vel],
        },
        vec=subj_vec,
    )

    return WorldState(
        src=src,
        t=t,
        frame=frame,
        entities=[subject],
        scene=scene,
    )
