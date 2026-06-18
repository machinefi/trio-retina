"""Render the world-model demo GIF — a RECOGNIZABLE, honest imagination rollout.

The goal of this GIF is instant legibility: a viewer should immediately read
**"objects are moving, and the model predicts where each one goes next, off
Retina's state."** So we re-skin the synthetic scene as a clean top-down road:
each tracked object is drawn as an unmistakable **car** sprite (rounded body +
windshield + a heading nub), labeled with its track id, with a short fading
**trail** behind it, the learned dynamics model's **predicted next path** drawn
AHEAD as an indigo dashed arrow, and the **actual** future drawn lighter/gray for
an honest side-by-side.

Nothing about the trajectories or the model is faked. The car sprites are purely
a *rendering* of the exact same recorded data the rest of the example uses: the
positions come from the seeded `_Mover` physics in `dataset.py`, the imagined
path is the genuine autoregressive rollout of the real 1-step `with_appearance`
transformer (trained here on the recorded `WorldState` sequences with real DINOv2
appearance vecs), and the gray "actual" path is the ground truth the scene takes.
The honest divergence on the curving object — where short kinematics can't see the
type-determined bank but appearance can — is shown as-is.

Visual language (matches the repo's new light look):
  * brand indigo (#4f46e5) = the learned model's PREDICTED next path (dashed, ahead)
  * gray                    = the ACTUAL future (lighter, for honest comparison)
  * a solid fading trail    = where the object has been

Run on the Mac Studio (MPS) with the [dynamics] extra:
    python examples/world_model/make_demo_gif.py \
        --data examples/world_model/data/sequences.json \
        --out media/world_model_demo.gif
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dynamics_model import (
    N_SLOTS,
    Normalizer,
    build_model,
    rollout,
    windows_from_sequences,
)

# ---------------------------------------------------------------------------
# Light, clean palette consistent with the repo's new look.
# ---------------------------------------------------------------------------
BG = (247, 248, 251)          # soft off-white page
ROAD = (228, 231, 238)        # pale asphalt
ROAD_EDGE = (205, 210, 220)   # lane border
LANE_DASH = (255, 255, 255)   # centre lane markings
INK = (28, 32, 44)            # primary text
SUBTLE = (120, 128, 144)      # secondary text
GRID = (214, 219, 228)

INDIGO = (79, 70, 229)        # #4f46e5 — the model's prediction
INDIGO_SOFT = (129, 122, 233)
GRAY = (150, 158, 172)        # actual future (honest comparison)
TRAIL = (168, 175, 190)       # where it has been

# Two distinct, calm car colours (one per track). Slot 0 = heavy, slot 1 = light.
CAR_COLORS = [(70, 90, 130), (210, 120, 70)]   # steel-blue, warm-amber
CAR_DARK = [(48, 62, 92), (150, 80, 42)]

LOOK_AHEAD = 7                # frames of actual future drawn ahead for comparison
PRED_DRAW = 6                 # steps of the model's prediction drawn (near-future)
SCENE = 256                   # native render coordinate space

# Viewport: zoom into the active region (the two travel bands) so the cars fill
# the frame instead of sitting in empty road. (x0, y0, x1, y1) in SCENE coords.
VIEW = (2.0, 48.0, 196.0, 200.0)
VW = VIEW[2] - VIEW[0]
VH = VIEW[3] - VIEW[1]

PANEL_W = 520                 # on-canvas panel width in px
SCALE = PANEL_W / VW          # SCENE px -> canvas px
S_W = PANEL_W
S_H = int(VH * SCALE)
PAD = 22
HEADER = 64
FOOTER = 42
WIDTH = S_W + 2 * PAD
HEIGHT = HEADER + S_H + FOOTER

CAR_LEN = 19.0                # car body length in SCENE px (scaled at draw time)
CAR_WID = 11.0                # car body width


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
    return PAD + (x - VIEW[0]) * SCALE


def _sy(y: float) -> float:
    return HEADER + (y - VIEW[1]) * SCALE


def _dashed_path(draw, pts, color, width=3, dash=10, gap=8):
    """Draw a dashed polyline through pts (list of (x,y) in canvas px)."""
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


def _arrowhead(draw, p0, p1, color, size=9):
    """Filled arrowhead at p1 pointing along p0->p1."""
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    a1 = ang + math.radians(150)
    a2 = ang - math.radians(150)
    draw.polygon(
        [p1,
         (p1[0] + size * math.cos(a1), p1[1] + size * math.sin(a1)),
         (p1[0] + size * math.cos(a2), p1[1] + size * math.sin(a2))],
        fill=color,
    )


def _rot(px, py, cx, cy, ca, sa):
    """Rotate (px,py) about (cx,cy) by angle given as (cos, sin)."""
    dx, dy = px - cx, py - cy
    return (cx + dx * ca - dy * sa, cy + dx * sa + dy * ca)


def _draw_car(draw, cx, cy, heading, body, dark):
    """Draw a recognizable top-down car centred at canvas (cx,cy), facing heading.

    A rounded body (oriented along travel) + a windshield band + a small heading
    nub at the nose, so direction of travel reads at a glance."""
    ca, sa = math.cos(heading), math.sin(heading)
    hl = CAR_LEN * SCALE / 2.0
    hw = CAR_WID * SCALE / 2.0

    # body as an oriented rounded rectangle, approximated by a filled polygon
    # with chamfered corners (looks like a car silhouette from above)
    chamf = hw * 0.55
    pts = [
        (cx + hl - chamf, cy - hw), (cx + hl, cy - hw + chamf),
        (cx + hl, cy + hw - chamf), (cx + hl - chamf, cy + hw),
        (cx - hl + chamf, cy + hw), (cx - hl, cy + hw - chamf),
        (cx - hl, cy - hw + chamf), (cx - hl + chamf, cy - hw),
    ]
    pts = [_rot(px, py, cx, cy, ca, sa) for px, py in pts]
    draw.polygon(pts, fill=body, outline=dark)
    # redraw outline thicker for crispness
    draw.line(pts + [pts[0]], fill=dark, width=max(1, int(SCALE)))

    # windshield: a light band toward the nose (front third)
    wsx0 = hl * 0.10
    wsx1 = hl * 0.55
    ws = [
        (cx + wsx0, cy - hw * 0.66), (cx + wsx1, cy - hw * 0.5),
        (cx + wsx1, cy + hw * 0.5), (cx + wsx0, cy + hw * 0.66),
    ]
    ws = [_rot(px, py, cx, cy, ca, sa) for px, py in ws]
    draw.polygon(ws, fill=(225, 232, 244))

    # rear window: a smaller darker band toward the tail
    rwx0 = -hl * 0.62
    rwx1 = -hl * 0.30
    rw = [
        (cx + rwx0, cy - hw * 0.58), (cx + rwx1, cy - hw * 0.52),
        (cx + rwx1, cy + hw * 0.52), (cx + rwx0, cy + hw * 0.58),
    ]
    rw = [_rot(px, py, cx, cy, ca, sa) for px, py in rw]
    draw.polygon(rw, fill=tuple(int(c * 0.78 + 220 * 0.22) for c in body))

    # bright headlight bar integrated into the nose, so direction reads clearly
    hlx = hl * 0.90
    head = [
        (cx + hlx, cy - hw * 0.72), (cx + hl, cy - hw * 0.5),
        (cx + hl, cy + hw * 0.5), (cx + hlx, cy + hw * 0.72),
    ]
    head = [_rot(px, py, cx, cy, ca, sa) for px, py in head]
    draw.polygon(head, fill=(255, 246, 210))


def draw_road(draw):
    """Paint a clean top-down two-lane road backdrop, clipped to the viewport."""
    x0, y0 = PAD, HEADER
    x1, y1 = PAD + S_W, HEADER + S_H
    draw.rectangle([x0, y0, x1, y1], fill=ROAD, outline=ROAD_EDGE, width=2)

    # The recorded scene has two horizontal travel bands:
    #   heavy (slot 0) ~ y 75,  light (slot 1) ~ y 148-176.
    # Lane separators framing those bands so the cars sit "in lanes".
    for yb in (48, 116, 196):
        yy = _sy(yb)
        if y0 < yy < y1:
            draw.line([(x0, yy), (x1, yy)], fill=ROAD_EDGE, width=2)
    # dashed centre markings inside each travel band
    for yb in (80, 158):
        yy = _sy(yb)
        if not (y0 < yy < y1):
            continue
        x = x0 + 12
        while x < x1 - 12:
            draw.line([(x, yy), (x + 18, yy)], fill=LANE_DASH, width=3)
            x += 32


def build_frames(positions, preds, k, headings):
    """Compose the animation frames.

    positions : (T, N, 2) ground-truth centroids per slot (nan if absent)
    preds     : dict cursor -> (H, N, 2) the model's predicted next-H centroids,
                rolled out FROM the actual WorldState at that cursor.
    headings  : (T, N) ground-truth heading angle per slot (for car orientation)
    """
    T = positions.shape[0]
    title = "Trio Retina — a learned dynamics model predicts each object's next move"
    # auto-fit the title so it never clips the canvas width
    title_size = 20
    while title_size > 12:
        f_try = _font(title_size, bold=True)
        d0 = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        if d0.textlength(title, font=f_try) <= WIDTH - 2 * PAD:
            break
        title_size -= 1
    f_title = _font(title_size, bold=True)
    f_sub = _font(13)
    f_leg = _font(12)
    f_lab = _font(12, bold=True)

    out_frames = []
    for cursor in range(k, T):
        canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(canvas, "RGBA")
        draw_road(draw)

        for s in range(N_SLOTS):
            body, dark = CAR_COLORS[s], CAR_DARK[s]

            # --- trail so far (solid, fading) ---
            trail = [positions[t, s] for t in range(0, cursor + 1)
                     if not np.isnan(positions[t, s, 0])]
            for i in range(1, len(trail)):
                a = i / len(trail)
                col = (*TRAIL, int(50 + 150 * a))
                draw.line([(_sx(trail[i - 1][0]), _sy(trail[i - 1][1])),
                           (_sx(trail[i][0]), _sy(trail[i][1]))],
                          fill=col, width=int(3 * SCALE / 2.0) + 1)

            # --- actual future ahead (gray solid line, for honest comparison) ---
            fut = [positions[t, s] for t in range(cursor, min(T, cursor + LOOK_AHEAD + 1))
                   if not np.isnan(positions[t, s, 0])]
            fpts = [(_sx(p[0]), _sy(p[1])) for p in fut]
            for i in range(1, len(fpts)):
                draw.line([fpts[i - 1], fpts[i]], fill=(*GRAY, 220),
                          width=int(3 * SCALE / 2.0) + 1)

            # --- predicted next moves ahead (indigo dashed arrow), from the model,
            #     rolled out from the ACTUAL WorldState at this cursor ---
            imag_pts = []
            if not np.isnan(positions[cursor, s, 0]):
                imag_pts.append((_sx(positions[cursor, s, 0]), _sy(positions[cursor, s, 1])))
            pred = preds.get(cursor)
            if pred is not None:
                for j in range(min(PRED_DRAW, pred.shape[0])):
                    ix, iy = pred[j, s, 0], pred[j, s, 1]
                    if np.isnan(ix) or not (-30 <= ix <= SCENE + 30 and -30 <= iy <= SCENE + 30):
                        continue
                    imag_pts.append((_sx(ix), _sy(iy)))
            if len(imag_pts) >= 2:
                _dashed_path(draw, imag_pts, (*INDIGO, 255),
                             width=int(3 * SCALE / 2.0) + 1)
                _arrowhead(draw, imag_pts[-2], imag_pts[-1], INDIGO, size=8 * SCALE / 2.0 + 4)

            # actual endpoint marker (hollow gray ring), drawn last so the honest
            # gap between predicted (indigo head) and actual (gray ring) is visible.
            if len(fpts) >= 2:
                ex, ey = fpts[-1]
                r = 4.0
                draw.ellipse([ex - r, ey - r, ex + r, ey + r],
                             fill=(255, 255, 255, 235), outline=(*GRAY, 255), width=2)

            # --- the car itself at the present position ---
            if not np.isnan(positions[cursor, s, 0]):
                cx, cy = _sx(positions[cursor, s, 0]), _sy(positions[cursor, s, 1])
                _draw_car(draw, cx, cy, headings[cursor, s], body, dark)
                # label: "car · #id" above the car
                lab = f"car  #{s + 1}"
                lw = draw.textlength(lab, font=f_lab)
                lx = cx - lw / 2
                ly = cy - CAR_WID * SCALE / 2 - 18
                # chip background for legibility
                draw.rounded_rectangle([lx - 6, ly - 2, lx + lw + 6, ly + 15],
                                       radius=7, fill=(255, 255, 255, 230),
                                       outline=(*body, 255), width=1)
                draw.text((lx, ly), lab, font=f_lab, fill=dark)

        # ---- header: one-line title + subtitle ----
        draw.text((PAD, 12), title, font=f_title, fill=INK)
        draw.text((PAD, 12 + title_size + 6),
                  "off one WorldState  ·  indigo = model's prediction, gray = actual",
                  font=f_sub, fill=SUBTLE)

        # ---- footer legend: ● now · — trail · → predicted · ··· actual ----
        ly = HEADER + S_H + 14
        lx = PAD
        # now
        draw.ellipse([lx, ly + 3, lx + 11, ly + 14], fill=CAR_COLORS[0], outline=CAR_DARK[0])
        lx += 16
        draw.text((lx, ly), "now", font=f_leg, fill=INK)
        lx += draw.textlength("now", font=f_leg) + 22
        # trail
        draw.line([(lx, ly + 8), (lx + 26, ly + 8)], fill=TRAIL, width=4)
        lx += 32
        draw.text((lx, ly), "trail", font=f_leg, fill=INK)
        lx += draw.textlength("trail", font=f_leg) + 22
        # predicted (model)
        _dashed_path(draw, [(lx, ly + 8), (lx + 30, ly + 8)], INDIGO, width=4, dash=8, gap=5)
        _arrowhead(draw, (lx + 22, ly + 8), (lx + 32, ly + 8), INDIGO, size=7)
        lx += 40
        draw.text((lx, ly), "predicted (model)", font=f_leg, fill=INK)
        lx += draw.textlength("predicted (model)", font=f_leg) + 22
        # actual
        draw.line([(lx, ly + 8), (lx + 26, ly + 8)], fill=GRAY, width=4)
        lx += 32
        draw.text((lx, ly), "actual", font=f_leg, fill=INK)

        out_frames.append(canvas.convert("RGB"))

    return out_frames


def to_positions(seq):
    """Extract (T,N,2) ground-truth centroids per stable slot for a sequence."""
    ids = sorted({e["id"] for st in seq for e in st["entities"]})
    slot = {eid: i % N_SLOTS for i, eid in enumerate(ids)}
    T = len(seq)
    pos = np.full((T, N_SLOTS, 2), np.nan, np.float32)
    for t, st in enumerate(seq):
        for e in st["entities"]:
            sl = slot.get(e["id"])
            if sl is None or sl >= N_SLOTS:
                continue
            pos[t, sl] = (e["cx"], e["cy"])
    return pos


def to_headings(positions):
    """Per-frame heading angle per slot from successive ground-truth positions.

    Smoothed over the trailing few frames so a car's nose points the way it is
    actually travelling (and doesn't jitter when nearly stationary)."""
    T, N, _ = positions.shape
    head = np.zeros((T, N), np.float32)
    for s in range(N):
        last = 0.0
        for t in range(T):
            # look back up to 3 frames for a stable direction
            v = None
            for back in (1, 2, 3):
                if t - back >= 0 and not np.isnan(positions[t, s, 0]) \
                        and not np.isnan(positions[t - back, s, 0]):
                    dx = positions[t, s, 0] - positions[t - back, s, 0]
                    dy = positions[t, s, 1] - positions[t - back, s, 1]
                    if math.hypot(dx, dy) > 0.5:
                        v = math.atan2(dy, dx)
                        break
            last = v if v is not None else last
            head[t, s] = last
    return head


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="examples/world_model/data/sequences.json")
    ap.add_argument("--out", default="media/world_model_demo.gif")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=240)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hold", type=int, default=12, help="frames to hold at the end")
    ap.add_argument("--ms", type=int, default=130, help="ms per frame")
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

    # Hold out the LAST sequence as the demo; train the 1-step model on the rest.
    demo_idx = len(sequences) - 1
    demo_seq = sequences[demo_idx]
    train_seqs = sequences[:demo_idx]

    norm = Normalizer(w, h)
    tr = windows_from_sequences(train_seqs, k=args.k, w=w, h=h, vec_dim=vec_dim,
                                vscale=norm.vscale, horizon=1)
    model = build_model(vec_dim=vec_dim, with_appearance=True, k=args.k)

    from dynamics import train  # reuse the exact trainer
    print("training 1-step with_appearance model for the demo rollout…")
    model = train(model, *tr, epochs=args.epochs, lr=args.lr, device=device, seed=args.seed)

    # Ground-truth tracks of the held-out demo seq.
    positions = to_positions(demo_seq)
    headings = to_headings(positions)
    T = positions.shape[0]
    ids = sorted({e["id"] for st in demo_seq for e in st["entities"]})
    slot = {eid: i % N_SLOTS for i, eid in enumerate(ids)}
    # last-known appearance vec per slot (carried into each rollout).
    last_vec = [np.zeros(vec_dim, np.float32) for _ in range(N_SLOTS)]
    for st in demo_seq:
        for e in st["entities"]:
            sl = slot.get(e["id"])
            if sl is not None and sl < N_SLOTS and e.get("vec") is not None:
                last_vec[sl] = np.asarray(e["vec"], np.float32)
    vecs = np.stack(last_vec)

    # At each cursor, roll the model forward LOOK_AHEAD steps FROM the actual
    # WorldState window ending at that cursor — the genuine "predict the next
    # moves off one state". This is the real model's output, fresh per frame.
    preds: dict[int, np.ndarray] = {}
    errs, errs_heavy, errs_light = [], [], []
    for cursor in range(args.k, T):
        win = [[None] * N_SLOTS for _ in range(args.k)]
        for ti in range(args.k):
            fi = cursor - (args.k - 1) + ti  # frames cursor-k+1 .. cursor
            for s in range(N_SLOTS):
                if not np.isnan(positions[fi, s, 0]):
                    win[ti][s] = (float(positions[fi, s, 0]), float(positions[fi, s, 1]))
        seed = {"pos": win, "vec": vecs}
        pr = rollout(model, seed, steps=LOOK_AHEAD, w=w, h=h)  # (H, N, 2)
        preds[cursor] = pr
        # honest near-future error: first step prediction vs actual next frame
        if cursor + 1 < T:
            for s in range(N_SLOTS):
                gt = positions[cursor + 1, s]
                if np.isnan(gt[0]) or np.isnan(pr[0, s, 0]):
                    continue
                e = math.hypot(pr[0, s, 0] - gt[0], pr[0, s, 1] - gt[1])
                errs.append(e)
                (errs_heavy if s == 0 else errs_light).append(e)
    print(f"demo seq (held-out): mean next-step predicted-vs-actual = "
          f"{np.mean(errs):.2f} px (car#1 {np.mean(errs_heavy):.2f} / "
          f"car#2 {np.mean(errs_light):.2f})")

    frames = build_frames(positions, preds, args.k, headings)
    frames = frames + [frames[-1]] * args.hold

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pal_frames = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
    pal_frames[0].save(
        args.out, save_all=True, append_images=pal_frames[1:],
        duration=args.ms, loop=0, optimize=True, disposal=2,
    )
    size_kb = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out} ({size_kb:.0f} KB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
