"""Unit tests for cheap frame gates."""

import numpy as np

from retina import MotionGate


def test_motion_gate_first_frame_always_looks():
    g = MotionGate(thresh=0.5)
    img = np.zeros((4, 4), dtype=np.uint8)
    assert g(img, 0.0) is True  # no previous frame -> always look


def test_motion_gate_near_identical_frames_skip():
    g = MotionGate(thresh=10.0)
    a = np.full((4, 4), 100, dtype=np.uint8)
    b = a.copy()
    b[0, 0] = 101  # one pixel off by 1 -> mean abs diff ~ 1/16, well under thresh
    g(a, 0.0)  # prime
    assert g(b, 1.0) is False


def test_motion_gate_large_change_looks():
    g = MotionGate(thresh=10.0)
    a = np.zeros((4, 4), dtype=np.uint8)
    b = np.full((4, 4), 255, dtype=np.uint8)  # huge change
    g(a, 0.0)  # prime
    assert g(b, 1.0) is True
