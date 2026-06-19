"""Heuristic, honest pass detection from the recorded Retina WorldState sequence.

Reads `passmap_states.json` (player tracks + per-frame ball signal + jersey HSV),
and produces:

  1. teams: k-means(k=2) on per-track median jersey HSV  -> {track_id: 0|1}
  2. possession: per frame, ball assigned to nearest player within a pixel
     threshold = the possessor (else "loose ball" for that frame).
  3. passes: when the possessor transfers A -> B with the ball in transit
     (A stable >= k frames, B stable >= k frames, ball travels a min distance),
     emit a Retina `pass.completed` Event. success = (B same team as A);
     A -> opponent = turnover / lost.

Everything is image-pixel space here; the renderer maps to pitch coords via an
approximate fixed homography. Emits real `retina.event/0.1` Events (custom
`pass.completed` type, pass payload in `ext`).
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ),
)

from retina.events import Event  # noqa: E402

SRC = "sports"


def hsv_to_xyz(hsv: list[float]) -> np.ndarray:
    """Map (H,S,V) to a cyclic-hue feature so red≈red across the 0/179 wrap.

    Hue is the discriminating channel for jersey colour; we weight it heavily
    and de-emphasise V (lighting varies a lot in this domed clip)."""
    h, s, v = hsv
    ang = h / 180.0 * 2 * math.pi
    return np.array(
        [math.cos(ang) * 2.0, math.sin(ang) * 2.0, s / 255.0 * 1.0, v / 255.0 * 0.3],
        np.float32,
    )


def assign_teams(states: list[dict], min_frames: int = 4) -> dict[str, int]:
    """k-means(k=2) on per-track median jersey colour. Returns {id: 0|1}.

    Only tracks seen in >= min_frames frames get a confident team; shorter
    tracks are assigned by nearest cluster centroid."""
    from sklearn.cluster import KMeans

    by_id: dict[str, list[list[float]]] = {}
    for st in states:
        for e in st["entities"]:
            by_id.setdefault(e["id"], []).append(e["hsv"])
    ids = sorted(by_id)
    feats = {i: hsv_to_xyz(list(np.median(np.asarray(by_id[i]), 0))) for i in ids}

    strong = [i for i in ids if len(by_id[i]) >= min_frames]
    if len(strong) < 2:
        strong = ids
    mat = np.stack([feats[i] for i in strong])
    km = KMeans(2, n_init=10, random_state=0).fit(mat)
    centers = km.cluster_centers_
    # stable ordering: team 0 = the larger cluster (the dominant white side here)
    lab = km.labels_
    if (lab == 1).sum() > (lab == 0).sum():
        lab = 1 - lab
        centers = centers[::-1]
    team = {i: int(lab[k]) for k, i in enumerate(strong)}
    # assign the rest by nearest centroid
    for i in ids:
        if i not in team:
            d0 = np.linalg.norm(feats[i] - centers[0])
            d1 = np.linalg.norm(feats[i] - centers[1])
            team[i] = 0 if d0 <= d1 else 1
    return team


def _ball_to_player(e: dict, bx: float, by: float) -> float:
    """Distance from the ball to a player's lower body.

    This low, ground-level clip means a player's bbox foot-point can be far below
    the ball even when the ball is at their feet (the dribbler's box runs to the
    image bottom). So we measure to the nearest point of the box's LOWER region
    (clamped to its bottom 40%), which tracks "ball at the player's feet"."""
    x1, y1, x2, y2 = e["bbox"]
    ylo = y1 + 0.60 * (y2 - y1)
    cx = min(max(bx, x1), x2)
    cy = min(max(by, ylo), y2)
    return math.hypot(cx - bx, cy - by)


def possession_series(states: list[dict], max_dist: float = 90.0) -> list[str | None]:
    """Per frame: id of the player whose lower body is nearest the ball, else None."""
    series: list[str | None] = []
    for st in states:
        ball = st.get("ball")
        if ball is None or not st["entities"]:
            series.append(None)
            continue
        bx, by = ball["cx"], ball["cy"]
        best, bestd = None, 1e9
        for e in st["entities"]:
            d = _ball_to_player(e, bx, by)
            if d < bestd:
                bestd, best = d, e["id"]
        series.append(best if bestd <= max_dist else None)
    return series


def _player_xy(st: dict, pid: str) -> tuple[float, float] | None:
    for e in st["entities"]:
        if e["id"] == pid:
            return (e["cx"], e["cy"])
    return None


def detect_passes(
    states: list[dict],
    teams: dict[str, int],
    poss: list[str | None],
    *,
    stable_k: int = 3,
    min_travel: float = 60.0,
    max_gap_frames: int = 30,
    reid_px: float = 70.0,
) -> list[Event]:
    """Detect A->B possession transfers and emit `pass.completed` Events.

    A transfer A->B counts as a pass when:
      * A was the (stable) possessor for >= stable_k of the frames before release,
      * B becomes the (stable) possessor for >= stable_k frames after receipt,
      * A != B, the receipt is within max_gap_frames of the release,
      * the ball/possessor travels >= min_travel px between release and receipt,
      * A and B are NOT the same physical player re-identified (their bodies are
        > reid_px apart at the transfer) — guards against the heavy track
        fragmentation this hard-panning clip causes.
    success = teams[B] == teams[A]; otherwise it is a turnover (lost ball).
    """
    # collapse possession into stable runs: [(pid, start_idx, end_idx), ...]
    runs: list[tuple[str, int, int]] = []
    for fi, pid in enumerate(poss):
        if pid is None:
            continue
        if runs and runs[-1][0] == pid and fi - runs[-1][2] <= 2:
            runs[-1] = (pid, runs[-1][1], fi)
        else:
            runs.append((pid, fi, fi))
    # keep only runs that are "stable" enough
    runs = [r for r in runs if (r[2] - r[1] + 1) >= stable_k]

    events: list[Event] = []
    for a, b in zip(runs, runs[1:], strict=False):
        a_id, _a0, a_end = a
        b_id, b0, _b1 = b
        if a_id == b_id:
            continue
        if b0 - a_end > max_gap_frames:
            continue
        st_from = states[a_end]
        st_to = states[b0]
        p_from = _player_xy(st_from, a_id)
        p_to = _player_xy(st_to, b_id)
        if p_from is None or p_to is None:
            continue
        # ball travel: prefer ball positions at release/receipt, else player pts
        bf = st_from.get("ball")
        bt = st_to.get("ball")
        fx, fy = (bf["cx"], bf["cy"]) if bf else p_from
        tx, ty = (bt["cx"], bt["cy"]) if bt else p_to
        travel = math.hypot(tx - fx, ty - fy)
        if travel < min_travel:
            continue
        # re-ID guard: if at the RELEASE frame B's box sits right on top of A's
        # foot point, B is almost certainly A re-identified after a track break,
        # not a distinct receiver. Skip it.
        b_at_release = _player_xy(st_from, b_id)
        if b_at_release is not None and math.hypot(
            b_at_release[0] - p_from[0], b_at_release[1] - p_from[1]
        ) < reid_px:
            continue
        # also require the receiver to be a genuinely sustained possessor
        # (b's run length already >= stable_k by construction of `runs`).
        team_a = teams.get(a_id, 0)
        success = teams.get(b_id, 0) == team_a
        ev = Event(
            type="pass.completed",
            t=round(st_from["t"], 3),
            src=SRC,
            ext={
                "from": a_id,
                "to": b_id,
                "from_xy": [round(fx, 1), round(fy, 1)],
                "to_xy": [round(tx, 1), round(ty, 1)],
                "team": team_a,
                "success": bool(success),
                "travel_px": round(travel, 1),
                "frame_from": st_from["frame_idx"],
                "frame_to": st_to["frame_idx"],
            },
        )
        events.append(ev)
    return events


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="examples/world_model/soccer/passmap/data/passmap_states.json")
    ap.add_argument("--out", default="examples/world_model/soccer/passmap/data/passes.jsonl")
    ap.add_argument("--max-dist", type=float, default=90.0)
    ap.add_argument("--stable-k", type=int, default=3)
    ap.add_argument("--min-travel", type=float, default=60.0)
    args = ap.parse_args()

    with open(args.data) as fp:
        data = json.load(fp)
    states = data["states"]
    print(
        f"data: {data['n_frames']} frames, {data['n_player_tracks']} tracks, "
        f"ball-detection rate {data['ball_hits']}/{data['n_frames']} "
        f"({data['ball_hits'] / data['n_frames']:.1%})"
    )

    teams = assign_teams(states)
    n0 = sum(1 for v in teams.values() if v == 0)
    print(f"teams (jersey-colour k-means): team0={n0}, team1={len(teams) - n0} tracks")

    poss = possession_series(states, max_dist=args.max_dist)
    n_poss = sum(1 for p in poss if p is not None)
    print(f"possession assigned in {n_poss}/{len(poss)} frames (nearest player <= {args.max_dist}px)")

    events = detect_passes(states, teams, poss, stable_k=args.stable_k, min_travel=args.min_travel)
    comp = [e for e in events if e.ext["success"]]
    turn = [e for e in events if not e.ext["success"]]
    print(f"passes detected: {len(events)}  (completed same-team {len(comp)}, turnover {len(turn)})")
    for tm in (0, 1):
        c = sum(1 for e in comp if e.ext["team"] == tm)
        tot = sum(1 for e in events if e.ext["team"] == tm)
        print(f"  team{tm}: {c}/{tot} completed")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fp:
        for e in events:
            fp.write(e.to_json() + "\n")
    print(f"\nwrote {args.out}  ({len(events)} events). Sample pass.completed lines:")
    for e in events[:6]:
        print("  " + e.to_json())


if __name__ == "__main__":
    main()
