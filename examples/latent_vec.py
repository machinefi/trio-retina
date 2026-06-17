"""The latent channel, populated by hand — runs with NO model and NO GPU.

Retina's state is **dual**: every entity carries a readable *symbolic* core AND an
optional model-tagged *latent* `vec` on the *same* record (see DESIGN.md / SPEC.md
"The latent channel"). The `Vec` type and the `entity.vec` slot are a real,
serializable interface, and built-in *producers* now ship to fill it automatically
(`DinoV2Embedder` per-object, `VJepa2Embedder` scene-level). This example shows
the interface is usable on its own by attaching your **own** embedding:

    track.user["vec"] = Vec(...).to_dict()   # any embedding you already have
        → flows into WorldState  →  entity.vec  →  serialize  →  round-trips

So if you have an embedder (a frozen backbone, a ReID head, anything), you can ride
the latent channel directly — or reach for a built-in producer
(`examples/world_model/multi_encoder.py`) when you'd rather not wire one yourself.

    python examples/latent_vec.py
"""

import numpy as np

from retina import IoUTracker, Vec, WorldState
from retina.detect import Detection
from retina.nodes import DetectorNode, TrackerNode
from retina.pipeline import Pipeline


class Walker:
    """One 'person' walking across the frame — one detection per call."""

    def __init__(self):
        self.f = 0

    def __call__(self, image):
        x, y = 60 + self.f * 8, 200 + self.f * 4
        self.f += 1
        return [Detection("person", (x - 25, y - 25, x + 25, y + 25), 0.9)]


def my_embedding(track) -> Vec:
    """Stand in for a real producer (ReID head / frozen V-JEPA ROI). Here we just
    hash the track's box into a tiny deterministic vector — the point is the
    *channel*, not the model. Swap this for your own embedder and nothing else
    changes."""
    x1, y1, x2, y2 = track.bbox
    rng = np.random.default_rng(int(x1 + y1) % 2**32)
    values = rng.standard_normal(8).round(3).tolist()
    return Vec(model="demo-reid/v0", dim=8, values=values)


def main() -> None:
    pipe = Pipeline(
        [DetectorNode(Walker()), TrackerNode(IoUTracker(min_hits=2))], source_id="cam"
    )

    entity = None
    for i in range(4):
        frame = pipe.process(np.zeros((400, 640, 3), np.uint8), float(i))
        # Attach YOUR embedding to each track's open `user` slot. `WorldState`
        # reads `track.user["vec"]` and carries it onto the entity automatically.
        for trk in frame.tracks:
            trk.user["vec"] = my_embedding(trk).to_dict()
        ws = WorldState.from_frame(frame)
        if ws.entities:
            entity = ws.entities[-1]

    assert entity is not None, "tracker should have confirmed the walker"
    print("entity (symbolic core + latent vec):")
    print(f"  id={entity.id}  type={entity.type}  bbox={tuple(round(v) for v in entity.bbox)}")
    print(f"  vec.model={entity.vec.model}  vec.dim={entity.vec.dim}")
    print(f"  vec.values={entity.vec.values}\n")

    # The dual state serializes as one record — symbol + latent, never collapsed —
    # and round-trips losslessly through JSON.
    blob = ws.to_json()
    print(f"serialized WorldState ({len(blob)} bytes):")
    print(f"  {blob}\n")

    rt = WorldState(**_load_entities(blob))
    e2 = rt.entities[-1]
    assert e2.vec is not None and e2.vec.values == entity.vec.values
    print("round-trip ok: the latent vec survived serialize → parse, attached to its entity.")


def _load_entities(blob: str) -> dict:
    """Rebuild a WorldState from its JSON, reconstructing nested Entity/Vec."""
    import json

    from retina import Entity

    d = json.loads(blob)
    ents = []
    for e in d.get("entities", []):
        vec = Vec(**e["vec"]) if "vec" in e else None
        ents.append(
            Entity(
                id=e["id"],
                type=e["type"],
                bbox=tuple(e["bbox"]) if "bbox" in e else None,
                conf=e.get("conf"),
                vec=vec,
            )
        )
    return {"src": d["src"], "t": d["t"], "frame": d.get("frame"), "entities": ents}


if __name__ == "__main__":
    main()
