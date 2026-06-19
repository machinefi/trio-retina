"""Render an Opta-style PASS MAP PNG from the detected `pass.completed` events.

Reads `passes.jsonl` (Retina events, image-pixel `from_xy`/`to_xy`), maps each
endpoint to metric pitch coordinates through an APPROXIMATE, FIXED image->pitch
homography, and draws:

  * a top-down pitch (outline, halfway line, both penalty + goal boxes, centre
    circle), Opta-ish light styling,
  * one arrow per detected pass: RED = completed same-team, GRAY = turnover/lost,
  * a legend with per-team completed/total counts and an honest disclaimer.

HONEST CAVEAT ON THE HOMOGRAPHY: `sports.mp4` is a low, ground-level cinematic
clip whose camera pans/translates hard across 14s — there is no single stable
tactical view. We calibrate ONE fixed homography by eye from the clearest pitch
view (the penalty-area/goal-line frame near the end) and apply it to every pass.
This is deliberately approximate: absolute pitch positions can be off by metres,
especially for passes recorded when the camera was framed very differently. The
map proves the chain (real detect -> Retina state -> pass.completed -> pitch map),
it is NOT an Opta-grade survey.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# StatsBomb-ish pitch, metres.
PITCH_L = 120.0
PITCH_W = 80.0

# Opta-ish palette
PITCH_LINE = "#9fb0a6"
PITCH_FILL = "#eef3f0"
RED = "#d1342f"      # completed, same team
GRAY = "#8a93a3"     # turnover / lost
INK = "#1b1f26"
MUTE = "#6b7280"


def fit_homography(src, dst):
    """Direct linear transform for a planar homography (cv2-free)."""
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
    hp = np.hstack([pts, np.ones((len(pts), 1))]) @ Hm.T
    return hp[:, :2] / hp[:, 2:3]


def build_homography(W: float, H: float):
    """APPROXIMATE fixed image->pitch homography for sports.mp4.

    Eyeballed from the end-of-clip penalty-area view (frame ~350): the goal sits
    at the left, the 16.5m box opens to the right, the goal line runs across the
    lower-left, a touchline recedes to the right. We map four image landmarks to
    the corresponding points on the StatsBomb-ish pitch (attacking the LEFT goal,
    so passes land in the left third). The mapping only needs to put the handful
    of detected passes onto a believable patch of pitch — it is not metric-exact.
    """
    # Source landmarks in image pixels (fractions of W,H), from the goal-mouth view.
    src = [
        (0.27 * W, 0.58 * H),   # near goal post / goal-line at left  -> goal-line, mid
        (0.55 * W, 0.58 * H),   # along the goal-line to the right     -> goal-line, lower
        (0.62 * W, 0.78 * H),   # near touchline foreground            -> ~penalty spot area
        (0.05 * W, 0.72 * H),   # left foreground on the goal side     -> goal-line, upper
    ]
    # Target on the pitch (metres), attacking the left goal (x small).
    dst = [
        (0.0, 40.0),    # goal-line centre
        (16.5, 52.0),   # box edge, lower
        (11.0, 40.0),   # penalty spot
        (0.0, 28.0),    # goal-line, upper
    ]
    return fit_homography(src, dst)


def draw_pitch(ax):
    """Top-down pitch in metres (length x in [0,120], width y in [0,80])."""
    ax.add_patch(mpatches.Rectangle((0, 0), PITCH_L, PITCH_W, facecolor=PITCH_FILL,
                                    edgecolor=PITCH_LINE, lw=1.8, zorder=0))
    lw = 1.6
    # halfway line + centre circle + spot
    ax.plot([PITCH_L / 2, PITCH_L / 2], [0, PITCH_W], color=PITCH_LINE, lw=lw, zorder=1)
    ax.add_patch(mpatches.Circle((PITCH_L / 2, PITCH_W / 2), 9.15, fill=False,
                                 edgecolor=PITCH_LINE, lw=lw, zorder=1))
    ax.add_patch(mpatches.Circle((PITCH_L / 2, PITCH_W / 2), 0.4, color=PITCH_LINE, zorder=1))
    # penalty + goal boxes both ends
    for end in (0, 1):
        x0 = 0 if end == 0 else PITCH_L
        sgn = 1 if end == 0 else -1
        ax.add_patch(mpatches.Rectangle((x0 if end == 0 else x0 - 16.5, 40 - 40.3 / 2),
                                        16.5, 40.3, fill=False, edgecolor=PITCH_LINE,
                                        lw=lw, zorder=1))
        ax.add_patch(mpatches.Rectangle((x0 if end == 0 else x0 - 5.5, 40 - 18.3 / 2),
                                        5.5, 18.3, fill=False, edgecolor=PITCH_LINE,
                                        lw=lw, zorder=1))
        ax.add_patch(mpatches.Circle((x0 + sgn * 11.0, 40), 0.4, color=PITCH_LINE, zorder=1))
    ax.set_xlim(-4, PITCH_L + 4)
    ax.set_ylim(-4, PITCH_W + 4)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="examples/world_model/soccer/passmap/data/passmap_states.json")
    ap.add_argument("--passes", default="examples/world_model/soccer/passmap/data/passes.jsonl")
    ap.add_argument("--out", default="examples/world_model/soccer/passmap/passmap_poc.png")
    ap.add_argument("--also", default="/Users/rc/Desktop/2026/playground-assets/passmap_poc.png")
    args = ap.parse_args()

    with open(args.data) as fp:
        meta = json.load(fp)
    W, H = float(meta["W"]), float(meta["H"])
    ball_rate = meta["ball_hits"] / max(1, meta["n_frames"])

    passes = []
    with open(args.passes) as fp:
        for line in fp:
            line = line.strip()
            if line:
                passes.append(json.loads(line))

    Hm = build_homography(W, H)

    fig, ax = plt.subplots(figsize=(12, 8.4), dpi=150)
    fig.patch.set_facecolor("#f7f8fb")
    draw_pitch(ax)

    # map + draw each pass
    counts = {0: [0, 0], 1: [0, 0]}  # team -> [completed, total]
    for p in passes:
        team = int(p["team"])
        ok = bool(p["success"])
        counts[team][1] += 1
        if ok:
            counts[team][0] += 1
        a = apply_h(Hm, p["from_xy"])[0]
        b = apply_h(Hm, p["to_xy"])[0]
        # clamp into the pitch so an approximate homography can't fling a point off
        a = (float(np.clip(a[0], 1, PITCH_L - 1)), float(np.clip(a[1], 1, PITCH_W - 1)))
        b = (float(np.clip(b[0], 1, PITCH_L - 1)), float(np.clip(b[1], 1, PITCH_W - 1)))
        color = RED if ok else GRAY
        ax.annotate(
            "", xy=b, xytext=a,
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.6,
                        "shrinkA": 4, "shrinkB": 4, "mutation_scale": 20},
            zorder=5,
        )
        ax.scatter([a[0]], [a[1]], s=46, color=color, edgecolor="white", lw=1.2, zorder=6)

    total = len(passes)
    comp_total = sum(c[0] for c in counts.values())
    ax.set_title(
        "Pass map (PoC) · Retina WorldState → pass.completed · 14s broadcast clip",
        fontsize=15, fontweight="bold", color=INK, pad=14,
    )

    # legend
    handles = [
        mpatches.Patch(color=RED, label=f"completed, same team ({comp_total})"),
        mpatches.Patch(color=GRAY, label=f"turnover / lost ({total - comp_total})"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=10,
                    bbox_to_anchor=(0.0, 1.0))
    leg.get_frame().set_edgecolor(PITCH_LINE)

    # per-team + honesty footer
    t0, t1 = counts[0], counts[1]
    sub = (
        f"team A (white side): {t0[0]}/{t0[1]} completed     "
        f"team B (blue side): {t1[0]}/{t1[1]} completed\n"
        f"real run: {meta['n_frames']} frames · ball-detection {ball_rate:.0%} "
        f"({meta['ball_hits']}/{meta['n_frames']}) · {meta['n_player_tracks']} player tracks · "
        f"{total} passes detected"
    )
    ax.text(0.5, -0.02, sub, transform=ax.transAxes, ha="center", va="top",
            fontsize=10, color=MUTE)
    ax.text(
        0.5, -0.115,
        "APPROXIMATE fixed homography (hard-panning ground-level clip) · generic COCO "
        "YOLO ball · nearest-player possession heuristic — positions ±metres, not Opta-grade",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
        color="#9aa3b2", style="italic",
    )

    fig.subplots_adjust(left=0.04, right=0.97, top=0.92, bottom=0.16)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, facecolor=fig.get_facecolor())
    print(f"wrote {args.out}")
    if args.also:
        os.makedirs(os.path.dirname(args.also) or ".", exist_ok=True)
        fig.savefig(args.also, facecolor=fig.get_facecolor())
        print(f"wrote {args.also}")
    print(
        f"passes={total} completed-same-team={comp_total} turnover={total - comp_total} "
        f"| teamA {t0[0]}/{t0[1]}  teamB {t1[0]}/{t1[1]}"
    )


if __name__ == "__main__":
    main()
