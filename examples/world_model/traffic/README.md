# Traffic speed radar — from a camera, in ~30 lines of state

A viral clip going around: a student wires up an old camera + a few API calls and
sells a "speed radar" to a city. The trick isn't a secret model — it's that
**speed measurement is a calibration problem sitting on top of tracked state.**
Once every vehicle is a tracked entity with a *metric* ground position, speed is
just `d(position)/dt`. This example builds exactly that on Retina.

```
YOLO ─► Detection ─► IoUTracker ─► WorldState ─► Entity.locus (metres) ─► km/h
 detect    interop      track       standard      ground homography      + events
```

The only non-trivial step is the **homography**: map image pixels to metres on
the road plane from four correspondences whose real distances you know (a lane is
~3.5 m wide, a dashed segment ~3 m). After that, `Entity.locus` carries metres
and the "radar" is a time-derivative that fits in one small file (`speed.py`).

## Run it now — no camera, no model, no GPU

```bash
cd examples/world_model/traffic
python synthetic_traffic.py
```

Fabricates a road of vehicles at *known* speeds, projects them through a fake
camera, runs them back through the **real** Retina state layer + speed estimator,
and checks the recovered km/h:

```
 id type     true  measured  error
  1 car      50.0     46.8  -3.2 km/h
  2 car      72.0     70.7  -1.3 km/h
  3 truck    33.0     32.5  -0.5 km/h
  4 car      61.0     62.0  +1.0 km/h

Speed-trap events (retina.event):
  {"type":"speed","t":1.5,"src":"synthetic_cam","id":2,"label":"car","kmh":70.7,"locus_m":[30.03,2.07]}
```

Each measurement is a real `retina.event` — the same wire format as `zone.enter`
/ `line.cross`, so it flows to any sink, dashboard, or LLM.

## Run it on real footage

```bash
# 1. calibrate: four image points ↔ their real-world metres (edit calib.example.json)
# 2. record: real YOLO + tracker + Retina state + speed
python record_traffic.py --clip road.mp4 --weights yolo11l.pt \
    --calib calib.example.json --out states.json
# 3. render the dashboard mp4
python render_traffic.py --clip road.mp4 --states states.json \
    --calib calib.example.json --out radar.mp4 --limit-kmh 60
```

`record_traffic.py` / `render_traffic.py` lazily import `cv2` / `ultralytics`;
the numpy-only core (`homography.py`, `speed.py`, `synthetic_traffic.py`) stays
importable with zero heavy deps, which is why the test suite runs in CI.

```bash
pytest test_traffic_demo.py -q     # 8 passed
```

## Honest scope

- **Analytics-grade, not enforcement-grade.** Single-camera speed is accurate to
  a few km/h at a well-calibrated mid-field trap — great for traffic intelligence
  (flow, over-speed flags, wrong-way, dwell), *not* a certified radar for writing
  tickets (that needs calibrated radar/lidar and legal metrology).
- **Accuracy degrades with range.** Near the top of the frame, perspective
  compresses many metres into few pixels, so detector jitter blows up. Measure at
  a mid-field trap where the geometry is well-conditioned — which is what the
  `SpeedEstimator` trap does.
- **Calibration is the whole game.** A fixed camera calibrates once; a moving/PTZ
  camera needs re-calibration (or per-frame road keypoints) — out of scope here.

## Why this lives in Retina

`speed` is a **domain verb built from primitives** — tracked entities + a metric
`locus` + a `line.cross`-style trap — so it stays in `examples/`, never in the
app-agnostic core. What the core contributes is the part worth standardizing: the
`Entity.locus` metric channel and one `WorldState`/`retina.event` contract every
sensor and consumer can share. Swap YOLO for any detector; the radar downstream
doesn't change.
