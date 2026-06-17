"""Numpy-only smoke test for the Phase-2 latent-dynamics example.

The heavy training (torch on MPS) is example-only and NOT exercised here. This
test just proves the numpy-only data path works on a bare checkout:

  * the seeded dataset generator is deterministic and reproduces byte-for-byte,
  * objects are tracked with stable ids and the two types stay separable,
  * `windows_from_sequences` produces correctly-shaped supervised tensors,
  * the constant-velocity baseline runs and returns a finite error,

all without importing torch (the transformer builder is never called here).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_WM = Path(__file__).resolve().parent.parent / "examples" / "world_model"


def _load(name: str):
    """Load an example module from examples/world_model/ by path."""
    if str(_WM) not in sys.path:
        sys.path.insert(0, str(_WM))
    spec = importlib.util.spec_from_file_location(name, _WM / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass forward-ref resolution (with
    # `from __future__ import annotations`) can find the module in sys.modules.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_dataset_is_deterministic_and_numpy_only():
    dataset = _load("dataset")
    a = dataset.generate(2, 10, seed=0, use_dino=False)
    b = dataset.generate(2, 10, seed=0, use_dino=False)
    # byte-for-byte reproducible under the same seed
    assert a == b
    # torch must NOT have been pulled in by the numpy-only path
    assert "torch" not in sys.modules


def test_objects_tracked_with_stable_ids_and_two_types():
    dataset = _load("dataset")
    data = dataset.generate(1, 12, seed=0, use_dino=False)
    seq = data["sequences"][0]
    types = {e["type"] for st in seq for e in st["entities"]}
    assert types == {"heavy", "light"}
    # both objects present and stably-id'd across the whole sequence
    ids = {e["id"] for st in seq for e in st["entities"]}
    assert len(ids) == 2
    for st in seq:
        assert len(st["entities"]) == 2


def test_windows_shapes_and_baseline():
    dataset = _load("dataset")
    dm = _load("dynamics_model")
    data = dataset.generate(3, 14, seed=1, use_dino=False)
    seqs = data["sequences"]
    w, h, vec_dim = float(data["W"]), float(data["H"]), int(data["vec_dim"])
    k = 6
    feat, vec, mask, target = dm.windows_from_sequences(
        seqs, k=k, w=w, h=h, vec_dim=vec_dim
    )
    s = feat.shape[0]
    assert feat.shape == (s, k, dm.N_SLOTS, 4)
    assert vec.shape == (s, k, dm.N_SLOTS, vec_dim)
    assert mask.shape == (s, dm.N_SLOTS)
    assert target.shape == (s, dm.N_SLOTS, 2)
    assert np.isfinite(feat).all()

    # eval module's constant-velocity baseline returns a finite pixel error
    dyn = _load("dynamics")
    err = dyn.constant_velocity_error(seqs, k, w, h)
    assert np.isfinite(err) and err > 0
    assert "torch" not in sys.modules


def test_appearance_vec_is_type_separable():
    """The stand-in appearance vec must carry type (so the ablation is real):
    same-type cosine > cross-type cosine."""
    dataset = _load("dataset")
    data = dataset.generate(1, 8, seed=2, use_dino=False)
    seq = data["sequences"][0]
    heavy = [np.asarray(e["vec"]) for st in seq for e in st["entities"] if e["type"] == "heavy"]
    light = [np.asarray(e["vec"]) for st in seq for e in st["entities"] if e["type"] == "light"]

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    same = cos(heavy[0], heavy[-1])
    cross = cos(heavy[0], light[0])
    assert same > cross


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
