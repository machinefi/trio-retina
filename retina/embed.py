"""Per-object latent producers — the models that *fill* the `vec` channel.

Retina's state is dual: every entity keeps a readable symbolic core AND an
optional model-tagged latent `vec` (see DESIGN.md / SPEC.md "The latent
channel"). `examples/latent_vec.py` shows the channel is usable by hand today.
This module ships the first *real* producer: `DinoV2Embedder`, a frozen DINOv2
backbone that crops each track's box and attaches a genuine self-supervised
embedding — no fake hash, no placeholder.

It's an enricher: `DinoV2Embedder()(frame) -> frame`, writing
`track.user["vec"] = Vec(...).to_dict()` so `WorldState.from_frame` carries it
onto `entity.vec` automatically. Drop it into a pipeline right after the tracker:

    DetectorNode(...) | TrackerNode() | DinoV2Embedder() | WorldStateNode()

Heavy deps (torch / transformers / pillow) are imported lazily on first call, so
the numpy-only core never gains a torch dependency just by importing `retina`.
`pip install 'trio-retina[dino]'` to enable it.
"""

from __future__ import annotations

from typing import Any

from .compose import Pipeable
from .worldstate import Vec

# size -> (HF model id, embedding dim)
_SIZES: dict[str, tuple[str, int]] = {
    "small": ("facebook/dinov2-small", 384),
    "base": ("facebook/dinov2-base", 768),
    "large": ("facebook/dinov2-large", 1024),
}


class DinoV2Embedder(Pipeable):
    """Frozen DINOv2 per-object embedder — the first real `vec` producer.

    Callable enricher: for each track it crops `frame.image[y1:y2, x1:x2]`,
    runs DINOv2 over all crops in one batched forward pass, and attaches the
    L2-normalized embedding as `track.user["vec"] = Vec(...).to_dict()`. From
    there `WorldState.from_frame` lifts it onto `entity.vec`.

    `size` picks the backbone: ``small`` (dim 384, default), ``base`` (768),
    ``large`` (1024). `device="auto"` selects mps → cuda → cpu. Set `bgr=True`
    for OpenCV frames (cv2 is BGR); synthetic / RGB frames keep the default
    False. Empty or out-of-bounds crops are skipped (clamped to image bounds).
    """

    def __init__(
        self,
        size: str = "small",
        *,
        device: str = "auto",
        normalize: bool = True,
        bgr: bool = False,
    ):
        if size not in _SIZES:
            raise ValueError(
                f"unknown size {size!r}; choose one of {sorted(_SIZES)}"
            )
        self.size = size
        self.model_id, self.dim = _SIZES[size]
        self.device_arg = device
        self.normalize = normalize
        self.bgr = bgr
        # Heavy objects are built lazily on first __call__ (see _lazy).
        self._processor = None
        self._model = None
        self._torch = None
        self.device: str | None = None

    def to_node(self):
        from .nodes import EnricherNode

        return EnricherNode(self)

    def _lazy(self) -> None:
        """Import torch/transformers/PIL and build the model on first use.

        Kept out of module import so `import retina` (and constructing this
        class) stays numpy-only; only an actual `__call__` pulls in torch."""
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as e:  # pragma: no cover - exercised only with extra
            raise ImportError(
                "DinoV2Embedder needs torch + transformers. "
                "Install with: pip install 'trio-retina[dino]'"
            ) from e

        if self.device_arg == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        else:
            device = self.device_arg

        self._torch = torch
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id).to(device).eval()
        self.device = device

    def __call__(self, frame: Any) -> Any:
        self._lazy()
        image = frame.image
        if image is None or not getattr(frame, "tracks", None):
            return frame

        h, w = image.shape[0], image.shape[1]
        crops = []
        targets = []  # tracks aligned with crops (1:1)
        for trk in frame.tracks:
            x1, y1, x2, y2 = trk.bbox
            # clamp to image bounds, integer pixel coords
            ix1 = max(0, min(int(x1), w))
            iy1 = max(0, min(int(y1), h))
            ix2 = max(0, min(int(x2), w))
            iy2 = max(0, min(int(y2), h))
            if ix2 <= ix1 or iy2 <= iy1:
                continue  # empty crop
            crop = image[iy1:iy2, ix1:ix2]
            if crop.size == 0:
                continue
            if self.bgr:
                crop = crop[:, :, ::-1]
            crops.append(crop)
            targets.append(trk)

        if not crops:
            return frame

        embeddings = self._embed(crops)
        for trk, emb in zip(targets, embeddings, strict=True):
            trk.user["vec"] = Vec(
                model=f"dinov2-{self.size}",
                dim=self.dim,
                values=emb.round(5).tolist(),
            ).to_dict()
        return frame

    def _embed(self, crops: list) -> list:
        """Run DINOv2 over a list of HxWx3 RGB crops → list of 1-D numpy arrays."""
        from PIL import Image

        torch = self._torch
        pil = [Image.fromarray(c.astype("uint8")) for c in crops]
        inputs = self._processor(images=pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model(**inputs)
        # pooler_output is the CLS-derived global descriptor; fall back to the
        # raw CLS token if a given config omits it.
        feats = getattr(out, "pooler_output", None)
        if feats is None:
            feats = out.last_hidden_state[:, 0]
        if self.normalize:
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        return list(feats.cpu().numpy())
