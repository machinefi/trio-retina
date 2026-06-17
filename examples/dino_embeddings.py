"""The first REAL latent producer: DINOv2 per-object embeddings on the `vec` channel.

`examples/latent_vec.py` rides the latent channel with a *hand-rolled* hash. This
one swaps in a genuine producer: `DinoV2Embedder` crops each tracked object and
runs a frozen DINOv2 backbone, attaching a real 384-d self-supervised embedding —

    DetectorNode | TrackerNode | DinoV2Embedder | WorldState.from_frame
        track.user["vec"] = Vec("dinov2-small", 384, values=[...])  (REAL)
        → flows into WorldState → entity.vec → serialize → round-trips

So the criticism "the latent channel has no producer" is dead: here it is, end to
end, with actual DINOv2 weights.

Needs the extra (pulls torch + transformers + pillow, downloads HF weights):

    pip install 'trio-retina[dino]'
    python examples/dino_embeddings.py
"""

import numpy as np

from retina import DinoV2Embedder, IoUTracker, WorldState
from retina.detect import Detection
from retina.nodes import DetectorNode, TrackerNode
from retina.pipeline import Pipeline


def _draw_person(image, cx, cy, color):
    """Paint a crude, distinctive 'person' blob so the two objects differ
    visually — enough for DINOv2 to produce different embeddings."""
    x1, y1, x2, y2 = cx - 24, cy - 48, cx + 24, cy + 48
    image[max(0, y1):y2, max(0, x1):x2] = color
    # a contrasting 'head' patch so the two identities are visually separable
    image[max(0, y1):y1 + 24, cx - 10:cx + 10] = (255 - color[0], 255 - color[1], 255 - color[2])
    return (x1, y1, x2, y2)


class ScriptedScene:
    """Two distinct 'people' walking across the frame; renders them onto the
    image AND returns their boxes, so the crops carry real, separable pixels."""

    def __init__(self):
        self.f = 0

    def __call__(self, image):
        dets = []
        # person A: warm color, moving right
        ax = 90 + self.f * 10
        box_a = _draw_person(image, ax, 180, (200, 60, 40))
        dets.append(Detection("person", box_a, 0.95))
        # person B: cool color, moving left
        bx = 520 - self.f * 10
        box_b = _draw_person(image, bx, 300, (40, 90, 210))
        dets.append(Detection("person", box_b, 0.92))
        self.f += 1
        return dets


def _cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> None:
    scene = ScriptedScene()
    embedder = DinoV2Embedder(size="small")  # 384-d, real DINOv2 weights
    pipe = Pipeline(
        [DetectorNode(scene), TrackerNode(IoUTracker(min_hits=2))], source_id="cam"
    )

    ws = None
    last_vecs: dict[str, list[float]] = {}
    history: dict[str, list[list[float]]] = {}
    for i in range(4):
        image = np.zeros((400, 640, 3), np.uint8)
        frame = pipe.process(image, float(i))
        embedder(frame)  # REAL DINOv2: fills track.user["vec"]
        ws = WorldState.from_frame(frame)
        for e in ws.entities:
            if e.vec is not None:
                last_vecs[e.id] = e.vec.values
                history.setdefault(e.id, []).append(e.vec.values)

    assert ws is not None and ws.entities, "tracker should have confirmed two people"

    print("entities (symbolic core + REAL DINOv2 latent vec):\n")
    for e in ws.entities:
        bbox = tuple(round(v) for v in e.bbox) if e.bbox else None
        print(f"  id={e.id}  type={e.type}  bbox={bbox}")
        if e.vec is not None:
            head = [round(v, 4) for v in e.vec.values[:6]]
            print(f"    vec.model={e.vec.model}  vec.dim={e.vec.dim}")
            print(f"    vec.values[:6]={head} ...\n")

    # Sanity check: real embeddings should cluster by identity.
    # Same object across frames > two different objects in one frame.
    ids = list(history)
    if len(ids) >= 2 and all(len(history[i]) >= 2 for i in ids[:2]):
        a, b = ids[0], ids[1]
        same_a = _cosine(history[a][0], history[a][-1])
        same_b = _cosine(history[b][0], history[b][-1])
        diff = _cosine(last_vecs[a], last_vecs[b])
        print("cosine sanity check (real embeddings cluster by identity):")
        print(f"  same object {a} across frames : {same_a:.4f}")
        print(f"  same object {b} across frames : {same_b:.4f}")
        print(f"  different objects {a} vs {b}  : {diff:.4f}")
        print(f"  same > different ? {min(same_a, same_b) > diff}\n")

    blob = ws.to_json()
    print(f"serialized WorldState ({len(blob)} bytes, real 384-d vecs ride inside):")
    print(f"  {blob[:240]} ...")


if __name__ == "__main__":
    main()
