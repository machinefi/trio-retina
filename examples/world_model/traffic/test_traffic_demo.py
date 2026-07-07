"""Smoke tests for the traffic speed demo — keep it actually runnable.

Run from this directory so the sibling modules import:

    cd examples/world_model/traffic && pytest test_traffic_demo.py

Everything here is numpy-only (no torch, no cv2, no YOLO), so the full suite
runs offline in CI. The real-footage path (`record_traffic.py`) is exercised by
hand, not here.
"""

from __future__ import annotations

import numpy as np

from homography import apply_h, fit_homography, foot_point
from speed import SpeedEstimator, estimate_speeds
from synthetic_traffic import IMAGE_QUAD, VEHICLES, WORLD_QUAD, calibration, make_states


def test_homography_roundtrip():
    # world -> pixel -> world recovers the calibration points
    G = fit_homography(WORLD_QUAD, IMAGE_QUAD)
    Hinv = fit_homography(IMAGE_QUAD, WORLD_QUAD)
    back = apply_h(Hinv, apply_h(G, WORLD_QUAD))
    assert np.allclose(back, WORLD_QUAD, atol=1e-6)


def test_foot_point_is_bottom_centre():
    assert foot_point((10, 20, 30, 80)) == (20.0, 80.0)


def test_calibration_maps_pixels_to_metres():
    calib = calibration()
    # a bbox whose foot point is exactly a known image quad corner -> its metres
    x, y = IMAGE_QUAD[0]
    X, Y = calib.to_metres((x - 5, y - 40, x + 5, y))
    assert np.allclose((X, Y), WORLD_QUAD[0], atol=1e-6)


def test_entities_carry_locus_and_serialize():
    states, _gt, _calib = make_states(seconds=1.0)
    ws = next(ws for _t, ws in states if ws.entities)
    ent = ws.entities[0]
    assert ent.locus is not None and len(ent.locus) == 2
    # locus survives the standard serialization as a list
    d = ent.to_dict()
    assert "locus" in d and isinstance(d["locus"], list)


def test_synthetic_speed_within_tolerance():
    # at the mid-field trap, recovered km/h tracks ground truth for every vehicle
    states, gt, _calib = make_states(seconds=4.0, noise_px=1.5, seed=0)
    events = estimate_speeds(states, src="cam", trap_x=30.0, window=6)
    measured = {ev.id: ev.ext["kmh"] for ev in events}
    assert len(measured) == len(VEHICLES)  # every car tripped the trap once
    for v in VEHICLES:
        assert abs(measured[int(v.vid)] - v.kmh) <= 5.0, (v.vid, measured[int(v.vid)], v.kmh)


def test_noiseless_speed_is_accurate():
    # with no detector jitter the trap reading is essentially exact
    states, gt, _calib = make_states(seconds=4.0, noise_px=0.0)
    events = estimate_speeds(states, trap_x=30.0, window=6)
    measured = {ev.id: ev.ext["kmh"] for ev in events}
    for v in VEHICLES:
        assert abs(measured[int(v.vid)] - v.kmh) <= 2.0


def test_speed_events_are_valid_retina_events():
    states, _gt, _calib = make_states(seconds=4.0)
    events = estimate_speeds(states, src="cam1", trap_x=30.0)
    assert events, "expected at least one speed-trap event"
    for ev in events:
        d = ev.to_json()
        assert '"type":"speed"' in d
        assert ev.src == "cam1"
        assert ev.ext["kmh"] > 0
        assert len(ev.ext["locus_m"]) == 2


def test_roi_gate_rejects_out_of_patch_detections():
    # a horizon-divergent detection lands at absurd metres; roi_m must drop it
    from retina import Entity, WorldState

    est = SpeedEstimator(src="cam", trap_x=40.0, roi_m=(-15.0, 95.0, -25.0, 45.0))
    good = 0
    for i in range(8):
        t = i * 0.1
        ents = [
            Entity(id="1", type="car", bbox=(0, 0, 4, 3), locus=(30 + i * 2.0, 5.0)),
            Entity(id="9", type="car", bbox=(0, 0, 1, 1), locus=(8228.0, -6637.0)),  # horizon garbage
        ]
        est.update(t, WorldState(src="cam", t=t, entities=ents))
        e_good = next(e for e in ents if e.id == "1")
        e_bad = next(e for e in ents if e.id == "9")
        assert "speed_kmh" not in e_bad.attrs  # never measured
        if "speed_kmh" in e_good.attrs:
            good += 1
            assert e_good.attrs["speed_kmh"] < 200  # sane
    assert good > 0


def test_trap_fires_once_per_vehicle():
    states, _gt, _calib = make_states(seconds=4.0)
    est = SpeedEstimator(src="cam", trap_x=30.0)
    fired = []
    for t, ws in states:
        fired.extend(est.update(t, ws))
    ids = [ev.id for ev in fired]
    assert len(ids) == len(set(ids))  # no double counting
