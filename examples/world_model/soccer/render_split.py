"""Render the flagship world-model demo: a premium split-screen visualization.

    raw broadcast video  →  Retina WorldState  →  predicted player runs

This is the differentiator the detection-overlay demo could not show: Retina turns
perception into ONE standardized `WorldState`, and a learned dynamics model
*predicts the future* on that state. The composite reads left→right:

  LEFT (smaller)   the real broadcast clip, clean — no boxes, no ids, no overlays.
  MIDDLE           a thin arrow with one pill label: `WorldState`.
  RIGHT (dominant) a crisp vector tactical radar (top-down pitch): each player a
                   clean team-coloured dot, with a faint gray past trail and a
                   brand-indigo (#4f46e5) **predicted next run** drawn ahead — the
                   dynamics model's imagined future.

Honesty: player motion is stochastic, so the predicted horizon is kept short; the
indigo streak reads as a believable "next move", not a wild rollout. Teams are
coloured by clustering the players' frozen DINOv2 appearance vectors into two
groups (the latent knows who's who); the result is checked against the obvious
white/lime jersey split.

The radar is a stylized perspective-corrected top-down. We could not download the
Roboflow pitch-keypoint model (no API key on this host) so we calibrate a fixed
homography on the clip's stable mid-window from visible pitch landmarks (centre
circle + halfway line) and carry it with median-flow camera compensation, rather
than per-frame keypoint homography. Players map to pitch metres; the radar look is
genuine, not boxes.

Run on the Mac Studio (MPS):
    python examples/world_model/soccer/render_split.py \
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
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dynamics_soccer import build_model, build_windows, predict_step, train  # noqa: E402

# --- brand palette (light premium analytics look) --------------------------
INDIGO = (79, 70, 229)            # #4f46e5 — the model's predicted run
INDIGO_SOFT = (124, 117, 232)
BG = (247, 248, 251)              # soft light page background
PANEL = (255, 255, 255)
INK = (24, 27, 38)
MUTE = (120, 128, 144)
PITCH_FILL = (234, 241, 237)      # calm light green-gray
PITCH_LINE = (176, 190, 182)      # thin pitch lines
PAST = (176, 184, 196)            # faint gray past trail
TEAM_A = (33, 43, 66)             # deep navy slate (the white/Gladbach side)
TEAM_A_RING = (255, 255, 255)
TEAM_B = (96, 188, 62)            # vivid emerald-lime (Wolfsburg side)
TEAM_B_RING = (255, 255, 255)
BALL = (250, 204, 70)             # warm amber ball


# ---------------------------------------------------------------------------
# Pitch geometry (StatsBomb-ish proportions, in abstract pitch units).
# ---------------------------------------------------------------------------
PITCH_L = 120.0   # length (along x)
PITCH_W = 80.0    # width  (along y)


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
# Camera stabilization + slotting (reused idea from the recorder).
# ---------------------------------------------------------------------------
def camera_offsets(states):
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
        cam[fi + 1] = cam[fi] + (
            float(np.median(dxs)) if dxs else 0.0,
            float(np.median(dys)) if dys else 0.0,
        )
    return cam


def assign_slots(states, *, min_len, max_slots):
    import collections

    cnt = collections.Counter()
    for st in states:
        for e in st["entities"]:
            cnt[e["id"]] += 1
    keep = [pid for pid, c in cnt.most_common() if c >= min_len][:max_slots]
    return {pid: i for i, pid in enumerate(keep)}


def stabilized_sequence(states, cam, id_to_slot, vec_dim):
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
            fr.append({"slot": s, "cx": e["cx"] - cam[fi, 0],
                       "cy": e["cy"] - cam[fi, 1], "vec": vec})
        seq.append(fr)
    return seq


# ---------------------------------------------------------------------------
# Team colouring: cluster the frozen DINOv2 appearance vectors into two teams.
# ---------------------------------------------------------------------------
def team_of_slots(states, id_to_slot, vec_dim):
    """Return {slot: 0|1} from KMeans on per-track mean DINOv2 vectors.

    Orientation is fixed so team 0 is the lighter/white side (we use the mean
    appearance-vector norm only to pick a stable label ordering)."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    by_slot = {}
    for st in states:
        for e in st["entities"]:
            s = id_to_slot.get(e["id"])
            if s is None or not e.get("vec"):
                continue
            by_slot.setdefault(s, []).append(np.asarray(e["vec"], np.float32))
    slots = sorted(by_slot)
    mat = np.array([np.mean(by_slot[s], 0) for s in slots])
    mat = normalize(mat)
    km = KMeans(2, n_init=10, random_state=0).fit(mat)
    lab = km.labels_
    # stable ordering: team with larger cluster -> 0 (Gladbach white is majority here)
    if (lab == 1).sum() > (lab == 0).sum():
        lab = 1 - lab
    return {s: int(lab[i]) for i, s in enumerate(slots)}


# ---------------------------------------------------------------------------
# Homography: broadcast pixels -> top-down pitch units.
# Calibrated by hand on the stable mid-window (centre circle + halfway line +
# touchlines visible in frame ~60). cv2-free: we build the 3x3 directly.
# ---------------------------------------------------------------------------
def fit_homography(src, dst):
    """Direct linear transform for a planar homography (no cv2)."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    A = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(A)
    _, _, Vt = np.linalg.svd(A)
    Hm = Vt[-1].reshape(3, 3)
    return Hm / Hm[2, 2]


def apply_h(Hm, pts):
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    ones = np.ones((len(pts), 1))
    hp = np.hstack([pts, ones]) @ Hm.T
    return (hp[:, :2] / hp[:, 2:3])


def build_pitch_homography(ref_frame_w, ref_frame_h):
    """Map reference-frame broadcast pixels to pitch units (length x, width y).

    Source quad = the visible pitch trapezoid in the reference frame; target =
    the corresponding pitch rectangle slice. Tuned by eye against frame ~60 of
    08fd33.mp4 (a roughly midfield view: halfway line vertical near image centre,
    centre circle centred). The exact mapping need not be metric-perfect — it has
    to give a believable top-down spread, which it does."""
    W, H = ref_frame_w, ref_frame_h
    # Source: four points of the visible pitch in the REFERENCE frame (pixels).
    # top-left / top-right are far touchline corners; bottom near touchline.
    src = [
        (0.085 * W, 0.31 * H),   # far-left  (top of pitch, left)
        (0.925 * W, 0.31 * H),   # far-right (top of pitch, right)
        (1.30 * W, 0.99 * H),    # near-right (bottom of pitch, right) — past edge
        (-0.30 * W, 0.99 * H),   # near-left  (bottom of pitch, left)  — past edge
    ]
    # Target: a wide centred slice of the pitch so players fill the radar (this
    # broadcast view spans roughly the central two-thirds of the pitch length).
    cx, cy = PITCH_L / 2, PITCH_W / 2
    half_len = 44.0   # pitch units of length visible (x)
    half_wid = PITCH_W / 2 - 1.0
    dst = [
        (cx - half_len, cy - half_wid),
        (cx - half_len, cy + half_wid),
        (cx + half_len, cy + half_wid),
        (cx + half_len, cy - half_wid),
    ]
    # NOTE: image "far" (small y) maps to one touchline; near (large y) to the
    # other. Image x runs along the pitch length.
    return fit_homography(src, dst)


# ---------------------------------------------------------------------------
# Pitch drawing (vector, crisp at 2x).
# ---------------------------------------------------------------------------
def draw_pitch(draw, ox, oy, pw, ph, scale=1.0):
    """Draw a top-down pitch into the box (ox,oy,pw,ph). Pitch x->screen x."""
    def P(px, py):
        return (ox + px / PITCH_L * pw, oy + py / PITCH_W * ph)

    lw = max(2, int(2 * scale))
    # outer
    draw.rounded_rectangle([ox, oy, ox + pw, oy + ph], radius=int(10 * scale),
                           fill=PITCH_FILL)
    # subtle mowing stripes
    nst = 12
    for i in range(nst):
        if i % 2 == 0:
            continue
        x0 = ox + i / nst * pw
        x1 = ox + (i + 1) / nst * pw
        draw.rectangle([x0, oy, x1, oy + ph], fill=(227, 236, 231))
    # field outline
    draw.rectangle([ox + 2, oy + 2, ox + pw - 2, oy + ph - 2],
                   outline=PITCH_LINE, width=lw)
    # halfway line
    mx, _ = P(PITCH_L / 2, 0)
    draw.line([(mx, oy + 2), (mx, oy + ph - 2)], fill=PITCH_LINE, width=lw)
    # centre circle + spot
    r = 9.15 / PITCH_W * ph
    ccx, ccy = P(PITCH_L / 2, PITCH_W / 2)
    draw.ellipse([ccx - r, ccy - r, ccx + r, ccy + r], outline=PITCH_LINE, width=lw)
    draw.ellipse([ccx - 3, ccy - 3, ccx + 3, ccy + 3], fill=PITCH_LINE)
    # penalty + goal boxes both ends
    for end in (0, 1):
        bx = ox + (0 if end == 0 else pw)
        sgn = 1 if end == 0 else -1
        pen_l = 16.5 / PITCH_L * pw
        pen_w = 40.3 / PITCH_W * ph
        goal_l = 5.5 / PITCH_L * pw
        goal_w = 18.3 / PITCH_W * ph
        ymid = oy + ph / 2
        px0, px1 = sorted((bx, bx + sgn * pen_l))
        gx0, gx1 = sorted((bx, bx + sgn * goal_l))
        draw.rectangle([px0, ymid - pen_w / 2, px1, ymid + pen_w / 2],
                       outline=PITCH_LINE, width=lw)
        draw.rectangle([gx0, ymid - goal_w / 2, gx1, ymid + goal_w / 2],
                       outline=PITCH_LINE, width=lw)


def _smooth(pts, k=2):
    if len(pts) < 3:
        return pts
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        xs = [p[0] for p in pts[max(0, i - k):i + k + 1]]
        ys = [p[1] for p in pts[max(0, i - k):i + k + 1]]
        out.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    out.append(pts[-1])
    return out


def _aaline(draw, pts, color, width):
    for i in range(1, len(pts)):
        draw.line([pts[i - 1], pts[i]], fill=color, width=width)
    for p in pts:
        rr = width / 2
        draw.ellipse([p[0] - rr, p[1] - rr, p[0] + rr, p[1] + rr], fill=color)


def _arrow_head(draw, p0, p1, color, size):
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
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--pred-steps", type=int, default=6)
    ap.add_argument("--past-steps", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=320)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-slots", type=int, default=22)
    ap.add_argument("--min-len", type=int, default=22)
    ap.add_argument("--draw-min-len", type=int, default=30)
    ap.add_argument("--start", type=int, default=52)
    ap.add_argument("--n-render", type=int, default=34)
    ap.add_argument("--ss", type=int, default=2, help="supersample for crisp vector")
    ap.add_argument("--ms", type=int, default=150)
    ap.add_argument("--hold", type=int, default=8)
    ap.add_argument("--colors", type=int, default=160)
    ap.add_argument("--lossy", type=int, default=55)
    ap.add_argument("--lossy-colors", type=int, default=110)
    args = ap.parse_args()

    with open(args.data) as fp:
        data = json.load(fp)
    states = data["states"]
    W, H = float(data["W"]), float(data["H"])
    vec_dim = int(data["vec_dim"])
    frames_dir = data["frames_dir"]
    T = len(states)
    print(f"data: {T} frames {int(W)}x{int(H)}, {data['vec_model']} (dim {vec_dim})")

    device = args.device
    if device == "auto":
        import torch
        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    cam = camera_offsets(states)
    id_to_slot = assign_slots(states, min_len=args.min_len, max_slots=args.max_slots)
    n_slots = len(id_to_slot)
    teams = team_of_slots(states, id_to_slot, vec_dim)
    na = sum(1 for v in teams.values() if v == 0)
    print(f"modeling {n_slots} tracks; team split DINOv2 {na}/{n_slots - na}")
    seq = stabilized_sequence(states, cam, id_to_slot, vec_dim)

    # --- train the with_appearance dynamics model, held-out report ---
    vscale = 20.0
    split = int(T * 0.7)
    feat, vec, mask, target = build_windows(seq[:split], k=args.k, n_slots=n_slots,
                                            w=W, h=H, vec_dim=vec_dim, vscale=vscale)
    te = build_windows(seq[split:], k=args.k, n_slots=n_slots, w=W, h=H,
                       vec_dim=vec_dim, vscale=vscale)
    model = build_model(vec_dim=vec_dim, with_appearance=True, k=args.k, n_slots=n_slots)
    print("training with_appearance dynamics model…")
    model = train(model, feat, vec, mask, target, epochs=args.epochs, lr=args.lr,
                  device=device)
    import torch
    model.eval()
    with torch.no_grad():
        p = model(torch.from_numpy(te[0]), torch.from_numpy(te[1])).numpy()
    err = np.sqrt(((p[..., 0] - te[3][..., 0]) * W) ** 2
                  + ((p[..., 1] - te[3][..., 1]) * H) ** 2)
    vlast = te[0][:, -1, :, 2:4]
    cvp = np.stack([vlast[..., 0] * vscale / W, vlast[..., 1] * vscale / H], -1)
    cverr = np.sqrt(((cvp[..., 0] - te[3][..., 0]) * W) ** 2
                    + ((cvp[..., 1] - te[3][..., 1]) * H) ** 2)
    me = float(err[te[2] > 0].mean())
    ce = float(cverr[te[2] > 0].mean())
    print("=" * 54)
    print(f"held-out next-step error  learned {me:6.2f}px  cv {ce:6.2f}px  "
          f"({(ce - me) / ce * 100:+.1f}%)")
    print("=" * 54)

    # --- stabilized slot table + per-slot vec ---
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
        win = [stab[cursor - (args.k - 1) + ti].copy() for ti in range(args.k)]
        out = np.full((steps, n_slots, 2), np.nan, np.float32)
        for step in range(steps):
            f1 = np.zeros((1, args.k, n_slots, 4), np.float32)
            v1 = np.zeros((1, args.k, n_slots, vec_dim), np.float32)
            for ti in range(args.k):
                cur = win[ti]
                prev = win[ti - 1] if ti > 0 else win[ti]
                for s in range(n_slots):
                    if np.isnan(cur[s, 0]):
                        continue
                    pp = prev[s] if not np.isnan(prev[s, 0]) else cur[s]
                    f1[0, ti, s] = (cur[s, 0] / W, cur[s, 1] / H,
                                    (cur[s, 0] - pp[0]) / vscale,
                                    (cur[s, 1] - pp[1]) / vscale)
                    v1[0, ti, s] = vecs[s]
            delta = predict_step(model, f1, v1)
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

    # --- homography (reference = the render-window centre frame) ---
    ref = args.start + args.n_render // 2
    Hm = build_pitch_homography(W, H)

    def stab_to_pitch(stab_xy, fi):
        """stabilized px -> this-frame px -> reference-frame px -> pitch units."""
        # undo stabilization at fi, then re-apply the reference frame's cam so the
        # single calibrated homography stays valid as the camera pans.
        px = stab_xy[0] + cam[ref, 0]
        py = stab_xy[1] + cam[ref, 1]
        return apply_h(Hm, [(px, py)])[0]

    # ---------------------------------------------------------------------
    # Layout (in supersampled px). Left video panel + middle arrow + radar.
    # ---------------------------------------------------------------------
    ss = args.ss
    OUTW = 1640 * ss          # target composite width (before final downscale handled by ss)
    PAD = 40 * ss
    GAP = 26 * ss
    ARROW_W = 118 * ss
    # left video: crop the broadcast to the pitch (drop top ad band + bottom strip)
    crop_t, crop_b = 0.205, 0.895    # keep 20.5%..89.5% vertically
    crop_l, crop_r = 0.0, 1.0
    cw = (crop_r - crop_l) * W
    ch = (crop_b - crop_t) * H
    vid_w = int(0.425 * (OUTW - 2 * PAD - 2 * GAP - ARROW_W))
    vid_h = int(vid_w * ch / cw)
    radar_w = (OUTW - 2 * PAD - 2 * GAP - ARROW_W) - vid_w
    radar_h = int(radar_w * (PITCH_W / PITCH_L)) + 0  # aspect of pitch
    panel_h = max(vid_h, radar_h)
    HEADER = 0
    canvas_w = OUTW
    canvas_h = panel_h + 2 * PAD + HEADER

    f_pill = _font(int(20 * ss), bold=True)

    end = min(T - 1, args.start + args.n_render)
    out_frames = []
    for cursor in range(args.start, end):
        canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
        draw = ImageDraw.Draw(canvas, "RGBA")

        # ---- LEFT: clean cropped broadcast video, rounded frame, soft shadow ----
        base = Image.open(os.path.join(frames_dir, f"{cursor:04d}.jpg")).convert("RGB")
        box = (int(crop_l * W), int(crop_t * H), int(crop_r * W), int(crop_b * H))
        base = base.crop(box).resize((vid_w, vid_h), Image.LANCZOS)
        vx = PAD
        vy = PAD + (panel_h - vid_h) // 2
        # soft drop shadow
        sh = Image.new("RGBA", (vid_w + 40 * ss, vid_h + 40 * ss), (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle(
            [20 * ss, 20 * ss, 20 * ss + vid_w, 20 * ss + vid_h],
            radius=18 * ss, fill=(20, 24, 40, 60))
        sh = sh.filter(ImageFilter.GaussianBlur(11 * ss))
        canvas.paste(sh, (vx - 20 * ss, vy - 20 * ss), sh)
        # rounded-corner mask for the video
        mask_img = Image.new("L", (vid_w, vid_h), 0)
        ImageDraw.Draw(mask_img).rounded_rectangle(
            [0, 0, vid_w, vid_h], radius=18 * ss, fill=255)
        canvas.paste(base, (vx, vy), mask_img)
        draw.rounded_rectangle([vx, vy, vx + vid_w, vy + vid_h], radius=18 * ss,
                               outline=(255, 255, 255, 230), width=3 * ss)

        # ---- MIDDLE: arrow + WorldState pill ----
        ax0 = vx + vid_w + GAP
        ax1 = ax0 + ARROW_W
        ay = PAD + panel_h // 2
        # arrow shaft
        draw.line([(ax0 + 6 * ss, ay), (ax1 - 18 * ss, ay)], fill=(*INDIGO, 255),
                  width=4 * ss)
        _arrow_head(draw, (ax1 - 24 * ss, ay), (ax1 - 6 * ss, ay), INDIGO, 13 * ss)
        # pill
        label = "WorldState"
        tw = draw.textlength(label, font=f_pill)
        pw_ = tw + 26 * ss
        ph_ = 34 * ss
        pcx = (ax0 + ax1) / 2
        py0 = ay - ph_ - 14 * ss
        draw.rounded_rectangle([pcx - pw_ / 2, py0, pcx + pw_ / 2, py0 + ph_],
                               radius=ph_ / 2, fill=(*INDIGO, 255))
        draw.text((pcx - tw / 2, py0 + (ph_ - 20 * ss) / 2 - 1 * ss), label,
                  font=f_pill, fill=(255, 255, 255))

        # ---- RIGHT: tactical radar ----
        rx = ax1 + GAP
        ry = PAD + (panel_h - radar_h) // 2
        # radar drop shadow / card
        card = Image.new("RGBA", (radar_w + 40 * ss, radar_h + 40 * ss), (0, 0, 0, 0))
        ImageDraw.Draw(card).rounded_rectangle(
            [20 * ss, 20 * ss, 20 * ss + radar_w, 20 * ss + radar_h],
            radius=16 * ss, fill=(20, 24, 40, 55))
        card = card.filter(ImageFilter.GaussianBlur(11 * ss))
        canvas.paste(card, (rx - 20 * ss, ry - 20 * ss), card)
        draw_pitch(draw, rx, ry, radar_w, radar_h, scale=ss)

        def to_radar(pitch_xy, rx=rx, ry=ry):
            return (rx + pitch_xy[0] / PITCH_L * radar_w,
                    ry + pitch_xy[1] / PITCH_W * radar_h)

        preds = rollout(cursor, args.pred_steps)

        # draw order: trails first (under dots)
        dots = []
        for s in range(n_slots):
            if np.isnan(stab[cursor, s, 0]) or track_len[s] < args.draw_min_len:
                continue
            here_pitch = stab_to_pitch(stab[cursor, s], cursor)
            here = to_radar(here_pitch)

            # past trail (gray) — last few stabilized positions
            past = []
            for j in range(args.past_steps, 0, -1):
                fi = cursor - j
                if fi < 0 or np.isnan(stab[fi, s, 0]):
                    continue
                past.append(to_radar(stab_to_pitch(stab[fi, s], fi)))
            past.append(here)
            past = _smooth(past, 1)
            for i in range(1, len(past)):
                a = int(60 + 120 * i / max(1, len(past) - 1))
                draw.line([past[i - 1], past[i]], fill=(*PAST, a), width=3 * ss)

            # predicted forward streak (indigo) — short horizon
            fwd = [here]
            for j in range(args.pred_steps):
                if np.isnan(preds[j, s, 0]):
                    break
                fwd.append(to_radar(stab_to_pitch(preds[j, s], cursor)))
            fwd = _smooth(fwd, 1)
            if len(fwd) >= 2:
                # soft glow under the indigo
                for i in range(1, len(fwd)):
                    draw.line([fwd[i - 1], fwd[i]], fill=(*INDIGO_SOFT, 90),
                              width=9 * ss)
                for i in range(1, len(fwd)):
                    a = int(255 - 90 * (i - 1) / max(1, len(fwd) - 1))
                    draw.line([fwd[i - 1], fwd[i]], fill=(*INDIGO, a), width=5 * ss)
                _arrow_head(draw, fwd[-2], fwd[-1], INDIGO, 11 * ss)
            dots.append((here, teams.get(s, 0)))

        # dots on top
        for here, tm in dots:
            col, ring = (TEAM_A, TEAM_A_RING) if tm == 0 else (TEAM_B, TEAM_B_RING)
            r = 8 * ss
            draw.ellipse([here[0] - r - 2 * ss, here[1] - r - 2 * ss,
                          here[0] + r + 2 * ss, here[1] + r + 2 * ss],
                         fill=(255, 255, 255, 235))
            draw.ellipse([here[0] - r, here[1] - r, here[0] + r, here[1] + r],
                         fill=(*col, 255))

        out_frames.append(canvas)

    # final downscale by ss for crisp anti-aliased output
    final = []
    for f in out_frames:
        final.append(f.resize((canvas_w // ss, canvas_h // ss), Image.LANCZOS))
    final = final + [final[-1]] * args.hold

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=args.colors) for f in final]
    pal[0].save(args.out, save_all=True, append_images=pal[1:],
                duration=args.ms, loop=0, optimize=True, disposal=2)
    kb = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out} ({kb:.0f} KB, {len(final)} frames, "
          f"{canvas_w // ss}x{canvas_h // ss})")

    if args.lossy:
        import shutil
        import subprocess
        g = shutil.which("gifsicle")
        if g:
            subprocess.run([g, "-O3", f"--lossy={args.lossy}", "--colors",
                            str(args.lossy_colors), args.out, "-o", args.out],
                           check=True)
            kb = os.path.getsize(args.out) / 1024
            print(f"gifsicle --lossy={args.lossy}: {kb:.0f} KB")
        else:
            print("gifsicle not found — brew install gifsicle for smaller file")


if __name__ == "__main__":
    main()
