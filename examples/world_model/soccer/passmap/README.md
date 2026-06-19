# Soccer pass-map PoC (Retina WorldState → `pass.completed`)

A **best-effort, honest** proof-of-concept: take a real soccer clip, route player
detections through the Retina state layer, detect possession transfers, emit
`pass.completed` Retina events, and render an Opta-style pass map.

```
frame (cv2) → YOLO (person + sports ball) → sv.Detections
  → Detection.from_supervision → DetectorNode | TrackerNode (Retina pipeline)
  → WorldState.from_frame  ─┐
                            ├→ possession (nearest player to ball)
  ball = per-frame signal ─┘   → A→B transfer → retina.event "pass.completed"
                                 → approximate fixed homography → pitch arrows (PNG)
```

## Run

```bash
# 1. detect + track + record (writes data/passmap_states.json, ~2.5 min on MPS)
python examples/world_model/soccer/passmap/record_passmap.py \
    --clip /path/to/sports.mp4 --weights yolo11m.pt --device mps --dump-frames

# 2. teams + possession + pass detection (writes data/passes.jsonl)
python examples/world_model/soccer/passmap/passes.py

# 3. render the pass map PNG
python examples/world_model/soccer/passmap/render_passmap.py
```

## What this PoC really is (and is not)

It proves the **chain** produces a real pass map from real perception — not a
full-match Opta survey. Measured on the bundled `sports.mp4` (1280×720, 25 fps,
352 frames, 14.1 s):

- ball-detection rate ~**78%** (273/352) — high here only because the clip is a
  close, ground-level dribble; generic COCO YOLO is weak on small far-away balls.
- **56** player tracks (heavy fragmentation — the camera pans/translates hard).
- **3** passes detected over 14 s (1 completed same-team, 2 turnovers). That is
  the right order of magnitude for a 1-v-1 dribble clip; no passes were invented.

### What's rough / why it's not Opta-grade

- 14 s single, low, ground-level cinematic view — not a tactical broadcast.
- Generic COCO YOLO ball/person detection (no roboflow soccer-specific models).
- **Approximate fixed homography** — the camera never holds still, so one
  calibrated homography puts passes on a *believable* patch of pitch, ±metres.
- Nearest-player (lower-body) possession heuristic; off-ball / off-camera passes
  are invisible; track fragmentation can split one player across many ids.
- Teams are jersey-colour k-means (k=2), not appearance-model identity.
