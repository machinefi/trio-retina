"""Swap the encoder, the state schema is constant.

The thesis of the latent channel (DESIGN.md / SPEC.md "The latent channel"): the
WorldState is a *fixed* dual-channel schema — a readable symbolic core plus an
optional model-tagged latent — and the encoder that fills the latent is
pluggable. This demo proves it by running the SAME pipeline three ways over the
SAME synthetic clip:

  1. symbolic only         — YOLO-style detections → tracks → entities, no latent
  2. + DinoV2Embedder      — per-object latent rides on each `entity.vec`
  3. + VJepa2Embedder      — scene-level latent rides on `ws.scene`

In all three the WorldState has the identical structure (src/t/frame/entities…);
only WHERE a latent rides — and which model produced it — changes. That is the
punchline: the state layer is the constant; the encoder is swapped underneath it.

Needs the extras for configs 2 and 3 (downloads HF weights):

    pip install 'trio-retina[dino]'    # config 2 (DINOv2 per-object)
    pip install 'trio-retina[vjepa]'   # config 3 (V-JEPA 2 scene)
    python examples/world_model/multi_encoder.py
"""

import numpy as np

from retina import DinoV2Embedder, IoUTracker, VJepa2Embedder, WorldState
from retina.detect import Detection
from retina.nodes import DetectorNode, TrackerNode
from retina.pipeline import Pipeline

CLIP_LEN = 8  # short clip so V-JEPA 2 fills its buffer quickly in the demo
W, H = 320, 240


def _background(f):
    """A non-blank, slowly drifting textured background so the encoders see real
    structure (black frames give degenerate features)."""
    yy, xx = np.mgrid[0:H, 0:W]
    r = (np.sin((xx + f * 7) / 23.0) * 90 + 110).astype(np.uint8)
    g = (np.cos((yy - f * 5) / 19.0) * 90 + 110).astype(np.uint8)
    b = (np.sin((xx + yy + f * 3) / 31.0) * 90 + 110).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _draw_person(image, cx, cy, color):
    """Paint a distinctive 'person' blob with a contrasting head patch."""
    x1, y1, x2, y2 = cx - 16, cy - 32, cx + 16, cy + 32
    image[max(0, y1):y2, max(0, x1):x2] = color
    image[max(0, y1):y1 + 16, cx - 7:cx + 7] = (255 - color[0], 255 - color[1], 255 - color[2])
    return (float(x1), float(y1), float(x2), float(y2))


class ScriptedDetector:
    """Two moving 'people' on the textured background, rendered onto the image
    and returned as YOLO-style detections."""

    def __init__(self):
        self.f = 0

    def __call__(self, image):
        dets = []
        ax = 50 + self.f * 8
        dets.append(Detection("person", _draw_person(image, ax, 120, (210, 70, 50)), 0.95))
        bx = 270 - self.f * 8
        dets.append(Detection("person", _draw_person(image, bx, 170, (50, 90, 220)), 0.92))
        self.f += 1
        return dets


def _frames(n):
    """A short synthetic clip: textured background + two moving people."""
    det = ScriptedDetector()
    out = []
    for i in range(n):
        img = _background(i)
        # detector both paints the people onto img AND returns their boxes
        dets = det(img)
        out.append((img, float(i), dets))
    return out


def _run(label, clip, *, dino=None, vjepa=None):
    """Run the same symbolic pipeline; optionally enrich with an encoder.

    Returns the LAST WorldState that has the relevant latent populated (so
    config 3 returns a frame where V-JEPA 2's rolling buffer has filled)."""
    det = ScriptedDetector()  # fresh detector to repaint identical frames
    pipe = Pipeline([DetectorNode(det), TrackerNode(IoUTracker(min_hits=2))], source_id="cam")
    result = None
    for img, t, _ in clip:
        frame = pipe.process(img.copy(), t)
        if dino is not None:
            dino(frame)         # fills track.user["vec"] → entity.vec
        if vjepa is not None:
            vjepa(frame)        # fills frame.user["scene"] → ws.scene
        ws = WorldState.from_frame(frame)
        # keep the most-enriched snapshot: scene fills late (after buffer fills)
        if vjepa is not None:
            if ws.scene is not None:
                result = ws
        else:
            result = ws
    return result


def _describe(ws):
    """The WorldState's structural shape — identical across all three configs."""
    d = ws.to_dict()
    keys = sorted(d.keys())
    n_ent = len(d.get("entities", []))
    ent_keys = sorted(d["entities"][0].keys()) if n_ent else []
    return keys, n_ent, ent_keys


def _print_config(label, ws, latent_note):
    keys, n_ent, ent_keys = _describe(ws)
    print(f"=== {label} ===")
    print(f"  WorldState keys : {keys}")
    print(f"  entities        : {n_ent}  (each entity keys: {ent_keys})")
    print(f"  latent rides at : {latent_note}")
    if ws.entities and ws.entities[0].vec is not None:
        v = ws.entities[0].vec
        print(f"    entity[0].vec : model={v.model} dim={v.dim} values[:4]={[round(x, 4) for x in v.values[:4]]} ...")
    if ws.scene is not None:
        s = ws.scene
        print(f"    scene         : model={s.model} dim={s.dim} values[:4]={[round(x, 4) for x in s.values[:4]]} ...")
    print()


def main() -> None:
    print("Swap the encoder, the state schema is constant.")
    print(f"(extras: [dino] for config 2, [vjepa] for config 3; clip_len={CLIP_LEN})\n")

    clip = _frames(CLIP_LEN)

    # 1) symbolic only — no latent producer at all
    ws_sym = _run("symbolic", clip)
    _print_config("config 1: symbolic only (no encoder)", ws_sym, "nowhere — symbolic core only")

    # 2) + DINOv2 per-object → entity.vec
    try:
        ws_dino = _run("dino", clip, dino=DinoV2Embedder(size="small"))
        _print_config("config 2: + DinoV2Embedder", ws_dino, "entity.vec (per-object, dim 384)")
    except ImportError as e:
        print(f"config 2 skipped (install 'trio-retina[dino]'): {e}\n")

    # 3) + V-JEPA 2 scene → ws.scene
    try:
        ws_vj = _run("vjepa", clip, vjepa=VJepa2Embedder(clip_len=CLIP_LEN))
        if ws_vj is None:
            print("config 3: V-JEPA 2 buffer never filled — increase clip length.\n")
        else:
            _print_config("config 3: + VJepa2Embedder", ws_vj, "ws.scene (scene-level)")
    except ImportError as e:
        print(f"config 3 skipped (install 'trio-retina[vjepa]'): {e}\n")

    print("same schema, swapped encoder — the state layer is the constant.")


if __name__ == "__main__":
    main()
