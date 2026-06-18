"""Train the soccer dynamics model and render the flagship world-model GIF.

Takes the recorded Retina `WorldState` sequence (real footage, real detection +
tracking + DINOv2 — see `record.py`), trains the small multi-player dynamics
transformer offline, and overlays — on the REAL frames — each well-tracked
player's marker + id, the model's **predicted next path** (brand indigo
`#4f46e5`, ahead of the player), and the player's **actual** next path (lighter
gray) for honest side-by-side comparison.

Player motion is stochastic, so the predictions are genuinely rough — we keep the
horizon short so the indigo arrow reads as "next move", report the true held-out
error, and show the gray actual path so the honest gap is visible.

Camera handling: the broadcast camera pans, so raw pixel velocity mixes camera
motion with player motion. We estimate the pan per frame as the robust median of
all players' displacement (median-flow) and predict in that **camera-stabilized**
frame, then map predictions back to each frame's pixels for the overlay — so the
arrows show where the *player* runs, not where the camera drifts.

Run on the Mac Studio (MPS):
    python examples/world_model/soccer/train_and_render.py \
        --data examples/world_model/soccer/data/soccer_states.json \
        --out media/world_model_soccer.gif
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dynamics_soccer import build_model, build_windows, predict_step, train  # noqa: E402

# --- brand palette ----------------------------------------------------------
INDIGO = (79, 70, 229)          # #4f46e5 — the model's prediction
INDIGO_LT = (129, 122, 233)
GRAY = (158, 165, 180)          # actual future (honest comparison)
WHITE = (245, 247, 250)
INK = (20, 22, 30)


def _font(size: int, bold: bool = False):
    cands = (
        ["/System/Library/Fonts/SFNSDisplay-Bold.otf",
         "/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Helvetica.ttc"]
        if bold else
        ["/System/Library/Fonts/SFNS.ttf",
         "/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Helvetica.ttc"]
    )
    for c in cands:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Stabilization + slotting.
# ---------------------------------------------------------------------------


def camera_offsets(states):
    """Cumulative camera offset per frame from robust median-flow.

    Returns `cam` (T,2): the cumulative camera translation at each frame, so that
    `stabilized = pixel - cam`. cam[0] = (0,0)."""
    T = len(states)
    pos = [{e["id"]: (e["cx"], e["cy"]) for e in st["entities"]} for st in states]
    cam = np.zeros((T, 2), np.float64)
    for fi in range(T - 1):
        dxs, dys = [], []
        for pid, p in pos[fi].items():
            q = pos[fi + 1].get(pid)
            if q is not None:
                dxs.append(q[0] - p[0])
                dys.append(q[1] - p[1])
        mdx = float(np.median(dxs)) if dxs else 0.0
        mdy = float(np.median(dys)) if dys else 0.0
        cam[fi + 1] = cam[fi] + (mdx, mdy)
    return cam


def assign_slots(states, *, min_len: int, max_slots: int):
    """Pick the longest-lived tracks and give them stable slot indices.

    Returns (id_to_slot, slot_to_id) for the top `max_slots` tracks with at least
    `min_len` frames — these are the ones clean enough to model and to draw."""
    import collections

    cnt = collections.Counter()
    for st in states:
        for e in st["entities"]:
            cnt[e["id"]] += 1
    keep = [pid for pid, c in cnt.most_common() if c >= min_len][:max_slots]
    id_to_slot = {pid: i for i, pid in enumerate(keep)}
    slot_to_id = {i: pid for pid, i in id_to_slot.items()}
    return id_to_slot, slot_to_id


def stabilized_sequence(states, cam, id_to_slot, vec_dim):
    """Build the camera-stabilized, slotted sequence for the model.

    Each frame -> list of {slot, cx, cy, vec} with cx,cy in stabilized coords."""
    seq = []
    for fi, st in enumerate(states):
        fr = []
        for e in st["entities"]:
            s = id_to_slot.get(e["id"])
            if s is None:
                continue
            vec = e.get("vec")
            if vec is None or len(vec) != vec_dim:
                vec = None
            fr.append({
                "slot": s,
                "cx": e["cx"] - cam[fi, 0],
                "cy": e["cy"] - cam[fi, 1],
                "vec": vec,
            })
        seq.append(fr)
    return seq


# ---------------------------------------------------------------------------
# Rendering helpers.
# ---------------------------------------------------------------------------


def _dashed(draw, pts, color, width=4, dash=11, gap=7):
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
            draw.line([(x1 + (x2 - x1) * t0, y1 + (y2 - y1) * t0),
                       (x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1)],
                      fill=color, width=width)


def _arrow(draw, p0, p1, color, size=12):
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    a1 = ang + math.radians(150)
    a2 = ang - math.radians(150)
    draw.polygon([p1,
                  (p1[0] + size * math.cos(a1), p1[1] + size * math.sin(a1)),
                  (p1[0] + size * math.cos(a2), p1[1] + size * math.sin(a2))],
                 fill=color)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="examples/world_model/soccer/data/soccer_states.json")
    ap.add_argument("--out", default="media/world_model_soccer.gif")
    ap.add_argument("--k", type=int, default=3, help="past-window length (frames)")
    ap.add_argument("--pred-steps", type=int, default=4,
                    help="frames of the model's next-move prediction drawn ahead")
    ap.add_argument("--look-ahead", type=int, default=4,
                    help="frames of actual future drawn (gray) for comparison")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-slots", type=int, default=14,
                    help="how many of the longest tracks to model & draw")
    ap.add_argument("--min-len", type=int, default=30)
    ap.add_argument("--draw-min-len", type=int, default=40,
                    help="only draw arrows for tracks at least this long (clean)")
    ap.add_argument("--start", type=int, default=52, help="first frame to render")
    ap.add_argument("--n-render", type=int, default=38, help="frames to render")
    ap.add_argument("--scale", type=float, default=0.40, help="output downscale factor")
    ap.add_argument("--ms", type=int, default=165)
    ap.add_argument("--hold", type=int, default=6)
    ap.add_argument("--colors", type=int, default=64, help="GIF palette size")
    ap.add_argument("--lossy", type=int, default=140,
                    help="gifsicle lossy level for the final squeeze (0 to skip)")
    ap.add_argument("--lossy-colors", type=int, default=36,
                    help="gifsicle palette size for the lossy squeeze")
    args = ap.parse_args()

    with open(args.data) as fp:
        data = json.load(fp)
    states = data["states"]
    W, H = float(data["W"]), float(data["H"])
    vec_dim = int(data["vec_dim"])
    frames_dir = data["frames_dir"]
    T = len(states)
    print(f"data: {T} frames {int(W)}x{int(H)}, vec={data['vec_model']} (dim {vec_dim})")

    device = args.device
    if device == "auto":
        import torch
        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    cam = camera_offsets(states)
    id_to_slot, slot_to_id = assign_slots(states, min_len=args.min_len,
                                           max_slots=args.max_slots)
    n_slots = len(id_to_slot)
    print(f"modeling {n_slots} long-lived player tracks (>= {args.min_len} frames)")
    seq = stabilized_sequence(states, cam, id_to_slot, vec_dim)

    # --- honest held-out eval: train on first 70%, test on last 30% of frames ---
    vscale = 20.0
    split = int(T * 0.7)
    feat, vec, mask, target = build_windows(
        seq[:split], k=args.k, n_slots=n_slots, w=W, h=H,
        vec_dim=vec_dim, vscale=vscale, horizon=1)
    te = build_windows(seq[split:], k=args.k, n_slots=n_slots, w=W, h=H,
                       vec_dim=vec_dim, vscale=vscale, horizon=1)
    print(f"train windows {feat.shape[0]}, test windows {te[0].shape[0]}")

    # constant-velocity baseline on the test windows (px)
    def cv_error(tarr, w, h):
        f, _, m, t = tarr
        # f[...,2:4] is last-step velocity (normalized by vscale); the CV
        # prediction delta over 1 frame equals that velocity, expressed as a
        # fraction of frame size: vx*vscale/w.
        vlast = f[:, -1, :, 2:4]
        pred = np.stack([vlast[..., 0] * vscale / w, vlast[..., 1] * vscale / h], -1)
        dpx = (pred[..., 0] - t[..., 0]) * w
        dpy = (pred[..., 1] - t[..., 1]) * h
        err = np.sqrt(dpx ** 2 + dpy ** 2)
        return float(err[m > 0].mean())

    cv = cv_error(te, W, H)

    model = build_model(vec_dim=vec_dim, with_appearance=True, k=args.k, n_slots=n_slots)
    print("training 1-step with_appearance dynamics model…")
    model = train(model, feat, vec, mask, target,
                  epochs=args.epochs, lr=args.lr, device=device, seed=args.seed)

    # model held-out error (px)
    import torch
    model.eval()
    with torch.no_grad():
        p = model(torch.from_numpy(te[0]), torch.from_numpy(te[1])).numpy()
    dpx = (p[..., 0] - te[3][..., 0]) * W
    dpy = (p[..., 1] - te[3][..., 1]) * H
    err = np.sqrt(dpx ** 2 + dpy ** 2)
    model_err = float(err[te[2] > 0].mean())
    print("=" * 58)
    print("HELD-OUT next-step player position error (px, lower=better)")
    print(f"  constant-velocity baseline      {cv:7.2f} px")
    print(f"  learned with_appearance         {model_err:7.2f} px")
    print(f"  improvement vs constant-velocity {(cv - model_err) / cv * 100:+5.1f}%")
    print("=" * 58)

    # --- per-frame predictions for rendering (camera-stabilized, then -> pixels) ---
    # slot table of stabilized positions per frame
    stab = np.full((T, n_slots, 2), np.nan, np.float32)
    last_vec = [np.zeros(vec_dim, np.float32) for _ in range(n_slots)]
    track_len = [0] * n_slots
    for fi, fr in enumerate(seq):
        for e in fr:
            s = e["slot"]
            stab[fi, s] = (e["cx"], e["cy"])
            track_len[s] += 1
            if e["vec"] is not None:
                last_vec[s] = np.asarray(e["vec"], np.float32)
    vecs = np.stack(last_vec)

    def rollout(cursor, steps):
        """Roll the model forward `steps` frames from the window ending at cursor.
        Returns (steps, n_slots, 2) stabilized predicted positions (nan if absent)."""
        win = [stab[cursor - (args.k - 1) + ti] for ti in range(args.k)]  # each (N,2)
        win = [w.copy() for w in win]
        out = np.full((steps, n_slots, 2), np.nan, np.float32)
        for step in range(steps):
            feat1 = np.zeros((1, args.k, n_slots, 4), np.float32)
            vec1 = np.zeros((1, args.k, n_slots, vec_dim), np.float32)
            for ti in range(args.k):
                cur = win[ti]
                prev = win[ti - 1] if ti > 0 else win[ti]
                for s in range(n_slots):
                    if np.isnan(cur[s, 0]):
                        continue
                    pp = prev[s] if not np.isnan(prev[s, 0]) else cur[s]
                    feat1[0, ti, s] = (cur[s, 0] / W, cur[s, 1] / H,
                                       (cur[s, 0] - pp[0]) / vscale,
                                       (cur[s, 1] - pp[1]) / vscale)
                    vec1[0, ti, s] = vecs[s]
            delta = predict_step(model, feat1, vec1)  # (N,2) normalized
            nxt = np.full((n_slots, 2), np.nan, np.float32)
            last = win[-1]
            for s in range(n_slots):
                if np.isnan(last[s, 0]):
                    continue
                nxt[s] = (last[s, 0] + delta[s, 0] * W, last[s, 1] + delta[s, 1] * H)
                out[step, s] = nxt[s]
            win.append(nxt)
            win.pop(0)
        return out

    # --- render ---
    title = "Trio Retina — a learned dynamics model predicts each player's next run"
    sub = "real match footage  ·  indigo = model's prediction  ·  gray = actual"

    sx = args.scale
    out_w = int(W * sx)
    canvas_w = out_w

    # Auto-fit the title to the canvas width so it never clips.
    _probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    title_size = 34
    while title_size > 13:
        if _probe.textlength(title, font=_font(title_size, bold=True)) <= canvas_w - 32:
            break
        title_size -= 1
    f_title = _font(title_size, bold=True)
    f_sub = _font(max(13, int(title_size * 0.62)))
    f_leg = _font(max(13, int(title_size * 0.60)))
    f_id = _font(max(12, int(title_size * 0.56)), bold=True)

    HEADER = title_size + int(title_size * 0.62) + 26
    FOOTER = int(title_size * 0.62) + 26
    canvas_h = int(H * sx) + HEADER + FOOTER

    end = min(T - 1, args.start + args.n_render)
    next_err_px = []
    out_frames = []
    for cursor in range(args.start, end):
        base = Image.open(os.path.join(frames_dir, f"{cursor:04d}.jpg")).convert("RGB")
        base = base.resize((out_w, int(H * sx)))
        canvas = Image.new("RGB", (canvas_w, canvas_h), WHITE)
        canvas.paste(base, (0, HEADER))
        draw = ImageDraw.Draw(canvas, "RGBA")

        preds = rollout(cursor, max(args.pred_steps, args.look_ahead))

        def to_px(stab_xy, fi):
            """stabilized -> this-frame pixel -> scaled canvas coords."""
            px = stab_xy[0] + cam[fi, 0]
            py = stab_xy[1] + cam[fi, 1]
            return (px * sx, py * sx + HEADER)

        for s in range(n_slots):
            if np.isnan(stab[cursor, s, 0]) or track_len[s] < args.draw_min_len:
                continue
            here = to_px(stab[cursor, s], cursor)

            # actual future (gray solid) — map each future stabilized pos to ITS frame
            fut = []
            for j in range(0, args.look_ahead + 1):
                fi = cursor + j
                if fi >= T or np.isnan(stab[fi, s, 0]):
                    break
                fut.append(to_px(stab[fi, s], fi))
            for i in range(1, len(fut)):
                # white halo under the gray actual path so it reads on grass
                draw.line([fut[i - 1], fut[i]], fill=(255, 255, 255, 200), width=7)
                draw.line([fut[i - 1], fut[i]], fill=(*GRAY, 255), width=4)
            if len(fut) >= 2:
                ex, ey = fut[-1]
                draw.ellipse([ex - 5, ey - 5, ex + 5, ey + 5],
                             fill=(255, 255, 255, 235), outline=(*GRAY, 255), width=2)

            # predicted next path (indigo arrow) — predictions are in the cursor's
            # stabilized frame; render them at the cursor's pixel mapping. A white
            # halo underneath makes the brand-indigo path pop against the pitch.
            pp = [here]
            for j in range(args.pred_steps):
                if np.isnan(preds[j, s, 0]):
                    break
                pp.append(to_px(preds[j, s], cursor))
            if len(pp) >= 2:
                for i in range(1, len(pp)):
                    draw.line([pp[i - 1], pp[i]], fill=(255, 255, 255, 210), width=8)
                for i in range(1, len(pp)):
                    draw.line([pp[i - 1], pp[i]], fill=(*INDIGO, 255), width=5)
                _arrow(draw, pp[-2], pp[-1], INDIGO, size=13)

            # player marker + id chip
            r = 7
            draw.ellipse([here[0] - r, here[1] - r, here[0] + r, here[1] + r],
                         fill=(*INDIGO, 255), outline=(255, 255, 255, 255), width=2)
            lab = f"#{s + 1}"
            lw = draw.textlength(lab, font=f_id)
            lx, ly = here[0] - lw / 2, here[1] - r - 20
            draw.rounded_rectangle([lx - 5, ly - 2, lx + lw + 5, ly + 19],
                                   radius=6, fill=(20, 22, 30, 205))
            draw.text((lx, ly), lab, font=f_id, fill=(235, 237, 245))

            # honest near-future error bookkeeping (1-step)
            if cursor + 1 < T and not np.isnan(stab[cursor + 1, s, 0]) \
                    and not np.isnan(preds[0, s, 0]):
                next_err_px.append(math.hypot(preds[0, s, 0] - stab[cursor + 1, s, 0],
                                              preds[0, s, 1] - stab[cursor + 1, s, 1]))

        # header
        draw.rectangle([0, 0, canvas_w, HEADER], fill=WHITE)
        draw.text((16, 10), title, font=f_title, fill=INK)
        draw.text((16, 12 + title_size + 2), sub, font=f_sub, fill=(110, 116, 130))

        # footer legend
        ly = canvas_h - FOOTER + 10
        lx = 20
        draw.ellipse([lx, ly + 2, lx + 13, ly + 15], fill=INDIGO, outline=(255, 255, 255))
        lx += 20
        draw.text((lx, ly), "player (now)", font=f_leg, fill=INK)
        lx += draw.textlength("player (now)", font=f_leg) + 26
        _dashed(draw, [(lx, ly + 9), (lx + 34, ly + 9)], INDIGO, width=4, dash=9, gap=5)
        _arrow(draw, (lx + 26, ly + 9), (lx + 36, ly + 9), INDIGO, size=8)
        lx += 46
        draw.text((lx, ly), "predicted next run (model)", font=f_leg, fill=INK)
        lx += draw.textlength("predicted next run (model)", font=f_leg) + 26
        draw.line([(lx, ly + 9), (lx + 30, ly + 9)], fill=GRAY, width=4)
        lx += 36
        draw.text((lx, ly), "actual", font=f_leg, fill=INK)

        out_frames.append(canvas.convert("RGB"))

    if next_err_px:
        print(f"rendered: mean drawn 1-step predicted-vs-actual (stabilized) = "
              f"{np.mean(next_err_px):.2f} px over {len(next_err_px)} player-frames")

    out_frames = out_frames + [out_frames[-1]] * args.hold
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=args.colors) for f in out_frames]
    pal[0].save(args.out, save_all=True, append_images=pal[1:],
                duration=args.ms, loop=0, optimize=True, disposal=2)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out} ({size_kb:.0f} KB, {len(out_frames)} frames, "
          f"{canvas_w}x{canvas_h})")

    # Real broadcast grass/crowd is palette-heavy, so squeeze the GIF with
    # gifsicle's lossy LZW if it's installed (keeps the overlays crisp while
    # shrinking the photographic background well under the size budget).
    if args.lossy:
        import shutil
        import subprocess

        gifsicle = shutil.which("gifsicle")
        if gifsicle:
            subprocess.run(
                [gifsicle, "-O3", f"--lossy={args.lossy}", "--colors",
                 str(args.lossy_colors), args.out, "-o", args.out],
                check=True,
            )
            size_kb = os.path.getsize(args.out) / 1024
            print(f"gifsicle --lossy={args.lossy} --colors {args.lossy_colors}: "
                  f"{size_kb:.0f} KB")
        else:
            print("gifsicle not found — skipping lossy squeeze "
                  "(brew install gifsicle for a smaller file)")


if __name__ == "__main__":
    main()
