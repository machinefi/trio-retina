"""Detection: the model-agnostic boundary.

Retina never imports a model. A *detector* is anything callable that maps a
frame to a list of `Detection`s. Bring YOLO, an RF-DETR, a Grounding-DINO
zero-shot prompt, or a frontier VLM behind an HTTP call — Retina doesn't care.
The optional `YoloDetector` is a convenience adapter, lazily importing
`ultralytics` only if you use it (keeps the core install tiny).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .compose import Pipeable
from .geometry import BBox


@dataclass(frozen=True, slots=True)
class Detection:
    """One object found in one frame."""

    label: str
    bbox: BBox  # (x1, y1, x2, y2) in pixels
    confidence: float = 1.0
    embedding: np.ndarray | None = None  # optional re-id appearance vector
    attrs: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        box = "(" + ",".join(f"{v:g}" for v in self.bbox) + ")"
        s = f"Detection(label={self.label!r} bbox={box} conf={self.confidence:.2f}"
        if self.embedding is not None:
            s += " +emb"
        return s + ")"

    @classmethod
    def from_supervision(
        cls,
        detections,
        class_names: dict[int, str] | list[str] | None = None,
    ) -> list[Detection]:
        """Ingest a Roboflow Supervision `sv.Detections` → `list[Detection]`.

        Supervision is the de-facto interop format ~20+ CV libraries convert
        into, so anyone already using it pipes straight into Retina's event
        layer. We never import `supervision` — the object is read by
        duck-typing: `.xyxy` (Nx4 [x1,y1,x2,y2]), `.confidence` (N or None),
        `.class_id` (N or None), `.data` (dict, may hold `"class_name"`).

        Label resolution, per row `i`: prefer `data["class_name"][i]`; else map
        `class_id[i]` through `class_names` (dict or list); else `str(class_id)`;
        else `""`. Missing `confidence` falls back to the `Detection` default."""
        xyxy = getattr(detections, "xyxy", None)
        if xyxy is None or len(xyxy) == 0:
            return []
        confidence = getattr(detections, "confidence", None)
        class_id = getattr(detections, "class_id", None)
        data = getattr(detections, "data", None) or {}
        class_name = data.get("class_name") if isinstance(data, dict) else None

        out: list[Detection] = []
        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = (float(v) for v in box)
            label = _resolve_label(i, class_name, class_id, class_names)
            kwargs: dict[str, Any] = {"label": label, "bbox": (x1, y1, x2, y2)}
            if confidence is not None:
                kwargs["confidence"] = float(confidence[i])
            out.append(cls(**kwargs))
        return out


def _resolve_label(
    i: int,
    class_name,
    class_id,
    class_names: dict[int, str] | list[str] | None,
) -> str:
    """Pick the best available label for row `i` of an `sv.Detections`."""
    if class_name is not None:
        try:
            return str(class_name[i])
        except (IndexError, KeyError):
            pass  # class_name shorter than xyxy -> fall through to class_id/""
    if class_id is None:
        return ""
    cid = int(class_id[i])
    if isinstance(class_names, dict):
        return str(class_names.get(cid, cid))
    if isinstance(class_names, (list, tuple)):
        if 0 <= cid < len(class_names):
            return str(class_names[cid])
    return str(cid)


@runtime_checkable
class Detector(Protocol):
    """Any object/callable that turns a frame into detections.

    A frame is an HxWx3 uint8 numpy array (or whatever your detector accepts —
    Retina just passes it through)."""

    def __call__(self, frame: np.ndarray) -> list[Detection]: ...


class CallableDetector(Pipeable):
    """Wrap a plain function as a Detector, optionally filtering classes /
    confidence. Lets you plug *any* model in one line."""

    def __init__(
        self,
        fn,
        *,
        classes: set[str] | None = None,
        min_confidence: float = 0.0,
    ):
        self._fn = fn
        self._classes = classes
        self._min_conf = min_confidence

    def to_node(self):
        from .nodes import DetectorNode

        return DetectorNode(self)

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        out = []
        for d in self._fn(frame):
            if d.confidence < self._min_conf:
                continue
            if self._classes is not None and d.label not in self._classes:
                continue
            out.append(d)
        return out


class YoloDetector(Pipeable):
    """Optional Ultralytics YOLO adapter. `pip install trio-retina[yolo]`.

    Loads any Ultralytics weights — YOLOv5/8/9/10/11/12, YOLO-World, RT-DETR —
    so swapping models is just a different weights string. Not imported unless
    you instantiate it, so the base package stays light."""

    def to_node(self):
        from .nodes import DetectorNode

        return DetectorNode(self)

    def __init__(
        self,
        weights: str = "yolo11n.pt",
        *,
        classes: set[str] | None = None,
        vocab: list[str] | None = None,
        min_confidence: float = 0.25,
        device: str | None = None,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover - exercised only with extra
            raise ImportError(
                "YoloDetector needs ultralytics. Install with: pip install 'trio-retina[yolo]'"
            ) from e
        self._model = YOLO(weights)
        # Open-vocabulary: with a YOLO-World weights file, `vocab` sets the
        # detectable classes from plain text — no training. (Ignored by closed
        # YOLO models, which raise; use it only with *-world weights.)
        if vocab is not None:
            self._model.set_classes(vocab)
            if classes is None:
                classes = set(vocab)
        self._device = device
        self._classes = classes
        self._min_conf = min_confidence

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            frame, verbose=False, conf=self._min_conf, device=self._device
        )
        out: list[Detection] = []
        for r in results:
            names = r.names
            for box in r.boxes:
                label = names[int(box.cls)]
                if self._classes is not None and label not in self._classes:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                out.append(
                    Detection(label=label, bbox=(x1, y1, x2, y2), confidence=float(box.conf))
                )
        return out


class VlmDetector(Pipeable):
    """Use ANY vision-language model as a detector.

    You pass a `client(image, prompt) -> iterable of dicts`, where each dict has
    `label`, `box` = [x1, y1, x2, y2] (pixels), and optional `score`. VlmDetector
    just maps that into `Detection`s — so Qwen-VL, Gemini, GPT-4o, Claude, or a
    local VLM all plug in behind the same seam. The client is yours (an
    OpenAI-compatible call, an HTTP request, etc.); keep grounding/JSON parsing
    there. A VLM can also be used as an EnricherNode/event source — see docs.
    """

    def __init__(
        self,
        client,
        prompt: str,
        *,
        classes: set[str] | None = None,
        min_confidence: float = 0.0,
    ):
        self._client = client
        self._prompt = prompt
        self._classes = classes
        self._min_conf = min_confidence

    def to_node(self):
        from .nodes import DetectorNode

        return DetectorNode(self)

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        out: list[Detection] = []
        for item in self._client(frame, self._prompt) or []:
            box = item.get("box")
            if box is None:
                continue
            score = float(item.get("score", 1.0))
            if score < self._min_conf:
                continue
            label = item.get("label", "object")
            if self._classes is not None and label not in self._classes:
                continue
            x1, y1, x2, y2 = box
            out.append(
                Detection(label=label, bbox=(float(x1), float(y1), float(x2), float(y2)), confidence=score)
            )
        return out


class GroundingDinoDetector(Pipeable):
    """Open-vocabulary detection from a text prompt via Grounding DINO (HF
    transformers). `pip install 'trio-retina[grounding]'`. Detects any classes you
    name — no training. Heavy (torch); not imported unless instantiated."""

    def __init__(
        self,
        classes: list[str],
        *,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        min_confidence: float = 0.3,
        device: str | None = None,
    ):
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as e:  # pragma: no cover - exercised only with extra
            raise ImportError(
                "GroundingDinoDetector needs transformers+torch. "
                "Install with: pip install 'trio-retina[grounding]'"
            ) from e
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        if device:
            self._model = self._model.to(device)
        self._device = device
        self._classes = classes
        self._prompt = ". ".join(c.lower() for c in classes) + "."
        self._min_conf = min_confidence

    def __call__(self, frame: np.ndarray) -> list[Detection]:  # pragma: no cover - needs model
        import torch
        from PIL import Image

        image = Image.fromarray(frame[:, :, ::-1])  # BGR->RGB
        inputs = self._processor(images=image, text=self._prompt, return_tensors="pt")
        if self._device:
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self._min_conf,
            target_sizes=[image.size[::-1]],
        )[0]
        out: list[Detection] = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            out.append(Detection(label=str(label), bbox=(x1, y1, x2, y2), confidence=float(score)))
        return out
