"""Render the world-model hero GIF — HONEST imagination rollout on the synthetic scene.

This composites the REAL trained dynamics model's output into a designed, looping
animation for the README. Nothing here is faked: the imagined trajectory is the
genuine autoregressive rollout of a 1-step `with_appearance` transformer trained on
the recorded `WorldState` sequences (real DINOv2 appearance vecs), and the "actual"
future is the ground-truth path the scene actually takes. The honest divergence on
the curving "light" object (where short kinematics can't see the type-determined
bank, but appearance can) is shown as-is.

Visual language matches the repo's hero caption convention:
  * gray    = baseline / ground-truth ("actual" future as it unfolds)
  * magenta = the learned model's imagination ("imagined" future, drawn ahead)

Pipeline:
  1. Load the recorded dataset (real DINOv2 vecs).
  2. Hold out one sequence; train a 1-step with-appearance model on the rest.
  3. Seed the model with the first K frames; imagine the whole future.
  4. Re-render the actual textured scene frames (reusing dataset.py painters) and
     composite the tracked trail + imagined (magenta) vs actual (gray) ahead.
  5. Write a small, palette-optimized, smooth-looping GIF via PIL.

Run on the Mac Studio (MPS) with the [dynamics] extra:
    python examples/world_model/make_hero_gif.py \
        --data examples/world_model/data/sequences.json \
        --out media/world_model_hero.gif
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import dataset as ds  # reuse the exact scene painters (_background, _paint_*, _Mover)
from dynamics_model import (
    N_SLOTS,
    Normalizer,
    build_model,
    rollout,
    windows_from_sequences,
)

# ---- design palette (brand-clean dark) ----
BG = (14, 16, 22)
PANEL = (20, 23, 31)
INK = (232, 236, 244)
SUBTLE = (120, 128, 144)
GRAY = (150, 158, 170)        # actual / ground-truth
MAGENTA = (232, 64, 196)      # the learned model's imagination
GRID = (34, 38, 48)

LOOK_AHEAD = 9                # frames of imagined future drawn ahead of the cursor
SCENE = 256                   # native render size
SCALE = 1.6                   # upscale factor for crispness
S = int(SCENE * SCALE)        # scaled scene px
PAD = 18
HEADER = 60
FOOTER = 40
WIDTH = S + 2 * PAD
HEIGHT = HEADER + S + FOOTER


def _font(size: int, bold: bool = False):
    cands = (
        ["/System/Library/Fonts/SFNSDisplay-Bold.otf", "/System/Library/Fonts/HelveticaNeue.ttc",
         "/System/Library/Fonts/Helvetica.ttc"]
        if bold else
        ["/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/HelveticaNeue.ttc",
         "/System/Library/Fonts/Helvetica.ttc"]
    )
    for c in cands:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _sx(x: float) -> float:
    return PAD + x * SCALE


def _sy(y: float) -> float:
    return HEADER + y * SCALE


def rebuild_movers(seed: int):
    """Reproduce the exact movers for a sequence seed (deterministic, same as dataset)."""
    rng = np.random.default_rng(seed)
    return ds._spawn(rng)


def _dashed_path(draw, pts, color, width=3, dash=9, gap=7):
    """Draw a dashed polyline through pts (list of (x,y))."""
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        seg = math.hypot(x2 - x1, y2 - y1)
        if seg < 1e-6:
            continue
        n = max(1, int(seg / (dash + gap)))
        for j in range(n + 1):
            t0 = (j * (dash + gap)) / seg
            t1 = min(1.0, t0 + dash / seg)
            if t0 >= 1.0:
                break
            draw.line(
                [(x1 + (x2 - x1) * t0, y1 + (y2 - y1) * t0),
                 (x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1)],
                fill=color, width=width,
            )


def _dot(draw, x, y, r, color, outline=None):
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=outline,
                 width=2 if outline else 0)


def build_frames(positions, imagined, k, w, h):
    """Compose the animation frames.

    positions : (T, N, 2) ground-truth centroids per slot (nan if absent)
    imagined  : (steps, N, 2) imagined centroids per slot, aligned to frames k..k+steps
    """
    T = positions.shape[0]
    # auto-fit the title so it never clips the canvas width
    t1, t2 = "Retina state", "learned dynamics imagines the future"
    title_size = 22
    while title_size > 12:
        f_try = _font(title_size, bold=True)
        d0 = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        w_title = (d0.textlength(t1, font=f_try) + 44
                   + d0.textlength(t2, font=f_try))
        if w_title <= WIDTH - 2 * PAD:
            break
        title_size -= 1
    f_title = _font(title_size, bold=True)
    f_sub = _font(max(10, title_size - 7))
    f_leg = _font(max(10, title_size - 8))

    out_frames = []
    # animate the "present" cursor from k .. T-1; at each cursor draw trail-so-far
    # (gray, solid), the actual future ahead (gray, faint), and the imagined future
    # ahead from the model (magenta, dashed). Honest: imagined diverges where real.
    for cursor in range(k, T):
        # Re-render the actual textured frame at `cursor`. The scene painters are
        # stateful (each call advances the movers), so replay from frame 0.
        movers = rebuild_movers(SEQ_SEED)
        img = None
        for fi in range(cursor + 1):
            bg = ds._background(fi).copy()
            for m in movers:
                (ds._paint_heavy if m.kind == "heavy" else ds._paint_light)(bg, m)
                m.step()
            img = bg
        scene = Image.fromarray(img, "RGB").resize((S, S), Image.BILINEAR)

        # dim the scene so overlays pop, and seat it on a dark panel
        canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
        scene = Image.blend(scene, Image.new("RGB", (S, S), PANEL), 0.32)
        canvas.paste(scene, (PAD, HEADER))
        draw = ImageDraw.Draw(canvas, "RGBA")

        # subtle frame border around the scene
        draw.rectangle([PAD, HEADER, PAD + S, HEADER + S], outline=GRID, width=2)

        for s in range(N_SLOTS):
            # --- tracked trail so far (gray, fading) ---
            trail = [(positions[t, s]) for t in range(0, cursor + 1)
                     if not np.isnan(positions[t, s, 0])]
            for i in range(1, len(trail)):
                a = i / len(trail)
                col = (*GRAY, int(40 + 150 * a))
                draw.line([(_sx(trail[i - 1][0]), _sy(trail[i - 1][1])),
                           (_sx(trail[i][0]), _sy(trail[i][1]))],
                          fill=col, width=int(2.5 * SCALE / 1.6) + 1)

            # --- actual future ahead (gray, faint solid), same horizon as imagined ---
            fut = [(positions[t, s]) for t in range(cursor, min(T, cursor + LOOK_AHEAD + 1))
                   if not np.isnan(positions[t, s, 0])]
            for i in range(1, len(fut)):
                draw.line([(_sx(fut[i - 1][0]), _sy(fut[i - 1][1])),
                           (_sx(fut[i][0]), _sy(fut[i][1]))],
                          fill=(*GRAY, 120), width=int(2 * SCALE / 1.6) + 1)
            for p in fut[1:]:
                _dot(draw, _sx(p[0]), _sy(p[1]), 2.2, (*GRAY, 150))

            # --- imagined future ahead (magenta dashed), from the present cursor ---
            # imagined[j] aligns to frame k+j; show the portion at/after cursor.
            imag_pts = []
            # anchor imagined at the actual present position for a clean handoff
            if not np.isnan(positions[cursor, s, 0]):
                imag_pts.append((_sx(positions[cursor, s, 0]), _sy(positions[cursor, s, 1])))
            for j in range(imagined.shape[0]):
                fi = k + j
                if fi <= cursor or fi > cursor + LOOK_AHEAD:
                    continue
                ix, iy = imagined[j, s, 0], imagined[j, s, 1]
                if np.isnan(ix) or not (-20 <= ix <= SCENE + 20 and -20 <= iy <= SCENE + 20):
                    continue
                imag_pts.append((_sx(ix), _sy(iy)))
            if len(imag_pts) >= 2:
                _dashed_path(draw, imag_pts, (*MAGENTA, 235),
                             width=int(2.5 * SCALE / 1.6) + 1)
                # arrowhead-ish endpoint marker
                ex, ey = imag_pts[-1]
                _dot(draw, ex, ey, 4.2, (*MAGENTA, 255), outline=(255, 255, 255))

            # --- the present position marker (where perception is now) ---
            if not np.isnan(positions[cursor, s, 0]):
                px, py = _sx(positions[cursor, s, 0]), _sy(positions[cursor, s, 1])
                _dot(draw, px, py, 5.5, (*INK, 255), outline=BG)

        # ---- header: title + subtitle ----
        # Draw "Retina state  ->  learned dynamics imagines the future" with a
        # vector arrow (so it renders regardless of font glyph coverage), kept
        # inside the canvas width.
        t1, t2 = "Retina state", "learned dynamics imagines the future"
        x = PAD
        ty = 11
        draw.text((x, ty), t1, font=f_title, fill=INK)
        x += draw.textlength(t1, font=f_title) + 12
        # arrow glyph
        ah = f_title.size
        ay = ty + ah * 0.55
        draw.line([(x, ay), (x + 16, ay)], fill=MAGENTA, width=2)
        draw.polygon([(x + 16, ay - 4), (x + 24, ay), (x + 16, ay + 4)], fill=MAGENTA)
        x += 32
        draw.text((x, ty), t2, font=f_title, fill=INK)
        draw.text((PAD, ty + ah + 4),
                  "one model-agnostic WorldState  •  small transformer rolls out the future",
                  font=f_sub, fill=SUBTLE)

        # ---- footer legend ----
        ly = HEADER + S + 13
        lx = PAD
        # imagined (magenta dashed)
        _dashed_path(draw, [(lx, ly + 6), (lx + 34, ly + 6)], MAGENTA, width=3, dash=7, gap=5)
        draw.text((lx + 42, ly), "imagined (learned model)", font=f_leg, fill=INK)
        lx2 = PAD + int(S * 0.55)
        draw.line([(lx2, ly + 6), (lx2 + 34, ly + 6)], fill=GRAY, width=3)
        draw.text((lx2 + 42, ly), "actual future", font=f_leg, fill=INK)

        out_frames.append(canvas.convert("RGB"))

    return out_frames


def to_positions(seq, w, h):
    """Extract (T,N,2) ground-truth centroids per stable slot for a sequence."""
    ids = sorted({e["id"] for st in seq for e in st["entities"]})
    slot = {eid: i % N_SLOTS for i, eid in enumerate(ids)}
    T = len(seq)
    pos = np.full((T, N_SLOTS, 2), np.nan, np.float32)
    for t, st in enumerate(seq):
        for e in st["entities"]:
            s = slot.get(e["id"])
            if s is None or s >= N_SLOTS:
                continue
            pos[t, s] = (e["cx"], e["cy"])
    return pos


def main() -> None:
    global SEQ_SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="examples/world_model/data/sequences.json")
    ap.add_argument("--out", default="media/world_model_hero.gif")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=240)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hold", type=int, default=14, help="frames to hold on the full rollout")
    ap.add_argument("--ms", type=int, default=120, help="ms per frame")
    args = ap.parse_args()

    with open(args.data) as fp:
        data = json.load(fp)
    sequences = data["sequences"]
    w, h, vec_dim = float(data["W"]), float(data["H"]), int(data["vec_dim"])
    print(f"dataset: {len(sequences)} seqs, vec={data['vec_model']} (dim {vec_dim})")

    device = args.device
    if device == "auto":
        import torch
        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Hold out the LAST sequence as the demo seq; train the 1-step model on the rest.
    demo_idx = len(sequences) - 1
    SEQ_SEED = int(data.get("seed", 0)) + demo_idx
    demo_seq = sequences[demo_idx]
    train_seqs = sequences[:demo_idx]

    norm = Normalizer(w, h)
    tr = windows_from_sequences(train_seqs, k=args.k, w=w, h=h, vec_dim=vec_dim,
                                vscale=norm.vscale, horizon=1)
    model = build_model(vec_dim=vec_dim, with_appearance=True, k=args.k)

    from dynamics import train  # reuse the exact trainer
    print("training 1-step with_appearance model for the hero rollout…")
    model = train(model, *tr, epochs=args.epochs, lr=args.lr, device=device, seed=args.seed)

    # Imagine the whole future of the held-out demo seq from its first K frames.
    positions = to_positions(demo_seq, w, h)
    T = positions.shape[0]
    steps = T - args.k
    # seed: first K frames' positions + last-known appearance vec per slot
    ids = sorted({e["id"] for st in demo_seq for e in st["entities"]})
    slot = {eid: i % N_SLOTS for i, eid in enumerate(ids)}
    last_vec = [np.zeros(vec_dim, np.float32) for _ in range(N_SLOTS)]
    pos_table = []
    for st in demo_seq:
        p = [None] * N_SLOTS
        for e in st["entities"]:
            s = slot.get(e["id"])
            if s is None or s >= N_SLOTS:
                continue
            p[s] = (e["cx"], e["cy"])
            if e.get("vec") is not None:
                last_vec[s] = np.asarray(e["vec"], np.float32)
        pos_table.append(p)
    seed = {"pos": [pos_table[i] for i in range(args.k)], "vec": np.stack(last_vec)}
    imagined = rollout(model, seed, steps=steps, w=w, h=h)  # (steps, N, 2)

    # honest divergence metric, for the report — over the near-future horizon that
    # the GIF actually draws (LOOK_AHEAD steps), per object type.
    errs_near, errs_heavy, errs_light = [], [], []
    for j in range(min(steps, LOOK_AHEAD)):
        fi = args.k + j
        for s in range(N_SLOTS):
            gt = positions[fi, s]
            if np.isnan(gt[0]) or np.isnan(imagined[j, s, 0]):
                continue
            e = math.hypot(imagined[j, s, 0] - gt[0], imagined[j, s, 1] - gt[1])
            errs_near.append(e)
            (errs_heavy if s == 0 else errs_light).append(e)
    print(f"hero seq (held-out, seed {SEQ_SEED}): mean imagined-vs-actual over "
          f"{min(steps, LOOK_AHEAD)} drawn steps = {np.mean(errs_near):.2f} px "
          f"(heavy {np.mean(errs_heavy):.2f} / light {np.mean(errs_light):.2f})")

    frames = build_frames(positions, imagined, args.k, w, h)
    # smooth loop: hold on the final fully-rolled-out frame
    frames = frames + [frames[-1]] * args.hold

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # palette-optimize for small size
    pal_frames = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    pal_frames[0].save(
        args.out, save_all=True, append_images=pal_frames[1:],
        duration=args.ms, loop=0, optimize=True, disposal=2,
    )
    size_kb = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out} ({size_kb:.0f} KB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
