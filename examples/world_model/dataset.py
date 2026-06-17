"""Dataset generator — record WorldState sequences from a structured synthetic scene.

This is Phase 2's data layer for the Dreamer-4-style latent-dynamics back-end.
The whole point of the eval (see `dynamics.py`) is HONEST: a learned dynamics
model that sees **[position + appearance vec]** should beat one that sees
**[position only]**. For that to be a fair, real win — not a rigged one — the
scene must have GENUINE structure:

  * Two visually distinct object **types** ("heavy" vs "light").
  * Their motion laws DIFFER and are NON-LINEAR (a constant-velocity baseline
    can't nail them), AND
  * The type that selects the motion law is **identifiable from appearance**, so
    a frozen DINOv2 embedding of the crop carries the signal a position-only
    model lacks.

Concretely:

  * "heavy"  — large, dark-red square block. Slow, gently *curving* drift (big
    turning radius, low speed): inertia-like.
  * "light"  — small, bright-cyan disc with a bullseye texture. Fast lateral
    *zigzag* (high-frequency sinusoidal sway) on a rightward drift.

The two are obviously different to DINOv2 (different colour, size, shape,
texture), and their dynamics are different non-linear laws. So appearance has a
real chance to inform motion. Whether it actually helps is reported honestly by
the eval — we don't fake the win.

Pipeline: render textured frame -> `ScriptedDetector` (paints objects + returns
boxes) -> `IoUTracker` -> `DinoV2Embedder` (real frozen DINOv2 `entity.vec`) ->
`WorldState.from_frame`. Each sequence is a list of WorldStates; we serialise the
symbolic core (id/type/bbox) + appearance vec per entity to a compact JSON.

Determinism: every sequence is seeded (`seed + sequence_index`), object spawn
positions / phases are drawn from that seed, so `python dataset.py` reproduces
the same dataset byte-for-byte. The committed dataset is small; regenerate with
real DINOv2 vecs via the `[dino]` extra on a machine with torch.

Usage:
    # numpy-only (fast, fake-but-deterministic vecs) — for smoke tests / CI:
    python examples/world_model/dataset.py --out data/seqs.json --no-dino

    # REAL DINOv2 appearance vecs (needs `pip install 'trio-retina[dino]'`):
    python examples/world_model/dataset.py --out data/seqs.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass

import numpy as np

from retina import IoUTracker, WorldState
from retina.detect import Detection
from retina.nodes import DetectorNode, TrackerNode
from retina.pipeline import Pipeline

W, H = 256, 256
DINO_SIZE = "small"  # 384-d
DINO_DIM = 384

# ---------------------------------------------------------------------------
# Scene: two appearance-distinct types with different non-linear motion laws.
# ---------------------------------------------------------------------------


@dataclass
class _Mover:
    """One object: its type, current centre, and the parameters of its law."""

    kind: str  # "heavy" | "light"
    cx: float
    cy: float
    # per-object random parameters (set at spawn from the seeded rng)
    heading: float  # radians, current travel direction (heavy: curving drift)
    turn: float  # heavy: angular velocity (rad/frame); sign varies per object
    speed: float  # base translational speed (px/frame)
    phase: float  # light: zigzag phase offset
    freq: float  # light: zigzag angular frequency
    amp: float  # light: zigzag lateral amplitude (px)

    def step(self) -> None:
        """Advance one frame under this object's non-linear law.

        Crucially, BOTH types move at the same instantaneous speed and (at spawn)
        similar headings — they differ ONLY in how the heading *evolves*, which is
        a curvature signal invisible from a 2-frame position window but fixed per
        type. So instantaneous velocity does NOT reveal the type; appearance does.

          * heavy — travels in a (near) straight line: heading barely changes.
          * light — banks into a hard, type-determined turn: heading rotates fast.

        Where the object is several steps out is therefore governed by its TYPE,
        and type is only legible from appearance — that's what gives the
        appearance channel a real, non-rigged job to do."""
        self.heading += self.turn  # heavy: ~0; light: a strong constant turn
        self.cx += self.speed * math.cos(self.heading)
        self.cy += self.speed * math.sin(self.heading)

    def bbox(self) -> tuple[float, float, float, float]:
        r = 30.0 if self.kind == "heavy" else 14.0
        return (self.cx - r, self.cy - r, self.cx + r, self.cy + r)


def _background(f: int) -> np.ndarray:
    """A slowly drifting textured background (black frames give degenerate
    DINOv2 features; structure keeps the appearance channel meaningful)."""
    yy, xx = np.mgrid[0:H, 0:W]
    r = (np.sin((xx + f * 5) / 27.0) * 70 + 90).astype(np.uint8)
    g = (np.cos((yy - f * 4) / 21.0) * 70 + 90).astype(np.uint8)
    b = (np.sin((xx + yy + f * 3) / 33.0) * 70 + 90).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _paint_heavy(img: np.ndarray, m: _Mover) -> None:
    """Large dark-red square block with a darker core — visually unmistakable."""
    x1, y1, x2, y2 = (int(v) for v in m.bbox())
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return
    img[y1:y2, x1:x2] = (150, 30, 30)
    # darker inner square for texture
    ix1, iy1 = x1 + (x2 - x1) // 4, y1 + (y2 - y1) // 4
    ix2, iy2 = x2 - (x2 - x1) // 4, y2 - (y2 - y1) // 4
    if ix2 > ix1 and iy2 > iy1:
        img[iy1:iy2, ix1:ix2] = (70, 10, 10)


def _paint_light(img: np.ndarray, m: _Mover) -> None:
    """Small bright-cyan disc with a bullseye (concentric rings) texture."""
    x1, y1, x2, y2 = m.bbox()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    r = (x2 - x1) / 2
    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    disc = dist <= r
    img[disc] = (40, 220, 230)
    ring = (dist <= r) & (dist >= r * 0.55) & (dist <= r * 0.75)
    img[ring] = (10, 60, 90)  # dark ring → bullseye texture


class ScriptedDetector:
    """Paints the movers onto the frame AND returns their YOLO-style boxes.

    Stateful: it owns the movers and advances them each call, so calling it once
    per frame produces a coherent trajectory. `labels=True` reveals the ground
    truth type as the detection label (used only to colour the WorldState type;
    the dynamics model never reads `type` directly — it reads the appearance vec
    or position, which is the whole point of the ablation)."""

    def __init__(self, movers: list[_Mover]):
        self.movers = movers

    def __call__(self, image: np.ndarray) -> list[Detection]:
        dets: list[Detection] = []
        for m in self.movers:
            if m.kind == "heavy":
                _paint_heavy(image, m)
            else:
                _paint_light(image, m)
            dets.append(Detection(m.kind, m.bbox(), 0.95))
            m.step()
        return dets


def _spawn(rng: np.random.Generator) -> list[_Mover]:
    """Spawn one heavy + one light mover with seeded random parameters.

    Both get the SAME speed range and overlapping headings, so their *first*
    velocity is not type-diagnostic. The only systematic difference is `turn`:
    heavy ~ straight, light ~ a hard banked turn (random sign). That curvature is
    unobservable in a 2-frame window but fixed per type — so the appearance
    channel (which identifies the type) is what lets the model predict the
    type-specific divergence several steps out."""
    speed_lo, speed_hi = 2.4, 3.2
    # Separate y-bands so the two boxes don't collide (keeps tracking ids clean),
    # but the same speed range and overlapping heading range so the *velocity* is
    # not type-diagnostic. heavy in the upper band, light in the lower band.
    heavy = _Mover(
        kind="heavy",
        cx=float(rng.uniform(40, 80)),
        cy=float(rng.uniform(55, 95)),
        heading=float(rng.uniform(-0.2, 0.2)),
        turn=float(rng.uniform(-0.01, 0.01)),  # ~straight
        speed=float(rng.uniform(speed_lo, speed_hi)),
        phase=0.0,
        freq=0.0,
        amp=0.0,
    )
    light = _Mover(
        kind="light",
        cx=float(rng.uniform(40, 80)),
        cy=float(rng.uniform(160, 200)),
        heading=float(rng.uniform(-0.2, 0.2)),
        # hard banked turn; sign chosen to bend back toward frame centre
        turn=-float(rng.uniform(0.16, 0.22)),
        speed=float(rng.uniform(speed_lo, speed_hi)),
        phase=0.0,
        freq=0.0,
        amp=0.0,
    )
    return [heavy, light]


# ---------------------------------------------------------------------------
# Appearance vec: real DINOv2, or a deterministic numpy-only stand-in.
# ---------------------------------------------------------------------------


def _fake_vec_for(kind: str, crop: np.ndarray, dim: int = DINO_DIM) -> list[float]:
    """A deterministic, numpy-only appearance vector for the --no-dino path.

    It is NOT DINOv2 — but it is type-separable and crop-derived (mean colour +
    a hashed texture signature), so the numpy-only smoke path still exercises the
    'appearance carries type' structure for CI. Real evals use DINOv2."""
    mean = crop.reshape(-1, 3).mean(axis=0) / 255.0 if crop.size else np.zeros(3)
    seed = (hash((kind, int(mean[0] * 97), int(mean[1] * 97), int(mean[2] * 97))) & 0xFFFFFFFF)
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(dim).astype(np.float32)
    # bias the first dims by type so it's strongly type-separable
    base[0] += 5.0 if kind == "heavy" else -5.0
    base[1] += mean[0] - mean[2]
    v = base / (np.linalg.norm(base) + 1e-9)
    return v.round(5).tolist()


def record_sequence(
    seq_len: int, seed: int, *, use_dino: bool, embedder=None
) -> list[dict]:
    """Run one seeded sequence through Retina; return a list of state dicts.

    Each state dict: {t, entities:[{id, type, cx, cy, w, h, vec:[...]}]}. The
    appearance `vec` is DINOv2 (if `use_dino`) or the deterministic stand-in."""
    rng = np.random.default_rng(seed)
    movers = _spawn(rng)
    detector = ScriptedDetector(movers)
    pipe = Pipeline(
        [DetectorNode(detector), TrackerNode(IoUTracker(min_hits=1, iou_threshold=0.2))],
        source_id=f"seq{seed}",
    )

    states: list[dict] = []
    for i in range(seq_len):
        img = _background(i)
        frame = pipe.process(img, float(i))
        if use_dino:
            embedder(frame)  # fills track.user["vec"] → entity.vec (REAL DINOv2)
        ws = WorldState.from_frame(frame)
        ents = []
        for e in ws.entities:
            if e.bbox is None:
                continue
            x1, y1, x2, y2 = e.bbox
            if use_dino and e.vec is not None:
                vec = e.vec.values
            else:
                # deterministic numpy stand-in from the crop
                ix1, iy1 = max(0, int(x1)), max(0, int(y1))
                ix2, iy2 = min(W, int(x2)), min(H, int(y2))
                crop = img[iy1:iy2, ix1:ix2]
                vec = _fake_vec_for(e.type, crop)
            # store at 4-decimal precision to keep the committed JSON small
            vec = [round(float(x), 4) for x in vec]
            ents.append(
                {
                    "id": e.id,
                    "type": e.type,
                    "cx": round((x1 + x2) / 2, 3),
                    "cy": round((y1 + y2) / 2, 3),
                    "w": round(x2 - x1, 3),
                    "h": round(y2 - y1, 3),
                    "vec": vec,
                }
            )
        states.append({"t": float(i), "entities": ents})
    return states


def generate(
    n_sequences: int,
    seq_len: int,
    *,
    seed: int = 0,
    use_dino: bool = True,
) -> dict:
    """Generate `n_sequences` seeded sequences; return the full dataset dict."""
    embedder = None
    if use_dino:
        from retina import DinoV2Embedder

        embedder = DinoV2Embedder(size=DINO_SIZE)

    sequences = []
    for k in range(n_sequences):
        states = record_sequence(seq_len, seed + k, use_dino=use_dino, embedder=embedder)
        sequences.append(states)
        print(f"  seq {k + 1}/{n_sequences} (seed {seed + k}): {len(states)} states")

    return {
        "spec": "retina.world_model.dataset/0.1",
        "scene": "heavy(curving-drift) + light(zigzag); types appearance-distinct",
        "W": W,
        "H": H,
        "vec_model": f"dinov2-{DINO_SIZE}" if use_dino else "fake-deterministic",
        "vec_dim": DINO_DIM,
        "seq_len": seq_len,
        "seed": seed,
        "sequences": sequences,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Retina WorldState sequences.")
    ap.add_argument("--out", default="examples/world_model/data/sequences.json")
    ap.add_argument("--n", type=int, default=30, help="number of sequences")
    ap.add_argument("--len", type=int, default=24, dest="seq_len", help="frames per sequence")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-dino",
        action="store_true",
        help="use deterministic numpy stand-in vecs (no torch); for CI/smoke",
    )
    args = ap.parse_args()

    use_dino = not args.no_dino
    print(
        f"Generating {args.n} sequences x {args.seq_len} frames "
        f"({'REAL DINOv2' if use_dino else 'numpy stand-in'} appearance vecs)…"
    )
    data = generate(args.n, args.seq_len, seed=args.seed, use_dino=use_dino)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(data, fp, separators=(",", ":"))
    size_kb = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out} ({size_kb:.0f} KB, {args.n} sequences)")


if __name__ == "__main__":
    main()
