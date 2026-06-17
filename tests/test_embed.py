"""Unit tests for `DinoV2Embedder` — the lazy-import contract.

These run numpy-only: constructing the embedder and importing `retina` must NOT
pull in torch (the heavy import happens only on `__call__`). We never call the
model here, so no torch/transformers are needed in the test env.
"""

import sys

import pytest

from retina import DinoV2Embedder


def test_import_retina_is_numpy_only():
    # Importing retina (and thus retina.embed) must not import torch.
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def test_construct_without_torch():
    # Building the embedder is allowed with no torch installed: nothing heavy
    # is imported until the first __call__ (which we don't make here).
    emb = DinoV2Embedder()
    assert emb.size == "small"
    assert emb.dim == 384
    assert emb.model_id == "facebook/dinov2-small"
    # still no torch pulled in by construction
    assert "torch" not in sys.modules


def test_sizes_map_to_dims():
    assert DinoV2Embedder("base").dim == 768
    assert DinoV2Embedder("large").dim == 1024
    assert DinoV2Embedder("small").dim == 384


def test_invalid_size_raises_value_error():
    with pytest.raises(ValueError):
        DinoV2Embedder("huge")


def test_to_node_wraps_in_enricher():
    from retina.nodes import EnricherNode

    node = DinoV2Embedder().to_node()
    assert isinstance(node, EnricherNode)


def test_call_without_torch_raises_helpful_importerror(monkeypatch):
    # Simulate torch missing: the lazy import on __call__ must raise the
    # install-hint ImportError, not a bare ModuleNotFoundError.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    emb = DinoV2Embedder()
    with pytest.raises(ImportError, match=r"trio-retina\[dino\]"):
        emb._lazy()
