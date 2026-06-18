# Cookbook

Task recipes, each a short runnable snippet that ends in a `retina.event` or a
`WorldState` — **not** a drawn frame. That's the point of Retina: a standardized
state + event stream the next layer can consume, not pixels on a screen.

The first recipes run on a bare `pip install trio-retina` (numpy only — no model,
no GPU, no video) using a stand-in detector, so you can paste and run them today.
No footage? `retina.sample_video()` returns a tiny generated clip for the
video-source path (see [recipe 6](#6-the-cli)).

A stand-in detector used by several recipes — one "person" walking across a dock:

```python
import numpy as np
from retina.detect import Detection

class Walker:
    """Stand-in model: one 'person' marching left→right, one step per frame."""
    def __init__(self, y=50):
        self._xs = list(range(0, 102, 6)); self._y = y
    def __call__(self, frame):
        x = self._xs.pop(0) if self._xs else 100
        return [Detection("person", (x - 10, self._y - 10, x + 10, self._y + 10), 0.9)]

frames = [(np.zeros((100, 100, 3), np.uint8), float(i)) for i in range(18)]
```

## 1. Zone intrusion → webhook

Fire `zone.enter` / `zone.dwell` / `zone.exit` when a person is in a restricted
zone, and POST each event to your endpoint (`WebhookSink`, stdlib urllib — no
`requests`). Swap `print` for a real `WebhookSink("https://your.app/ingest")`.

```python
from retina import Retina, Zone, ZoneRule, IoUTracker

dock = Zone("dock", [(40, 0), (60, 0), (60, 100), (40, 100)])
cam = Retina(
    source_id="cam_01",
    detector=Walker(),
    tracker=IoUTracker(min_hits=2),
    rules=[ZoneRule(dock, classes={"person"}, dwell_s=2.0)],
)

for event in cam.run(frames):
    print(event.to_json())
    # {"type":"zone.enter","t":7.0,"src":"cam_01","id":1,"label":"person","zone":"dock",...}
    # {"type":"zone.dwell","t":7.0,...,"zone":"dock","dur":2.0,...}
    # {"type":"zone.exit","t":7.0,...,"zone":"dock","dur":3.0,...}
```

To push instead of print, wire a sink — every event goes out as JSON:

```python
from retina import WebhookSink
cam = Retina(..., sinks=[WebhookSink("https://your.app/ingest")])
for _ in cam.run(frames):
    pass  # events are POSTed as they happen
```

## 2. Count people / line-crossing

A tripwire emits `line.cross` (with a `dir`), and `CountRule` emits
`count.threshold` the moment the live count crosses your threshold. Both need
*tracked* input — they're meaningless without object identity.

```python
from retina import Retina, Line, LineRule, CountRule, IoUTracker

tripwire = Line("door", (50, 0), (50, 100))
cam = Retina(
    source_id="cam_01",
    detector=Walker(),
    tracker=IoUTracker(min_hits=2),
    rules=[
        LineRule(tripwire, classes={"person"}),       # → line.cross {"dir":"a_to_b"}
        CountRule(threshold=1, classes={"person"}),   # → count.threshold {"n":1}
    ],
)
for event in cam.run(frames):
    print(event.to_json())
    # {"type":"count.threshold","t":1.0,"src":"cam_01","n":1,...,"threshold":1,"cmp":">="}
    # {"type":"line.cross","t":9.0,...,"zone":"door","dir":"a_to_b",...}
```

## 3. Ingest an existing Supervision `sv.Detections`

Already running Roboflow Supervision? `Detection.from_supervision` ingests an
`sv.Detections` straight into Retina's event layer — Retina never imports
`supervision` (it duck-types `.xyxy` / `.confidence` / `.class_id` / `.data`), so
anything that converts to Supervision pipes in. Here a tiny fake stands in for a
real `sv.Detections`:

```python
from types import SimpleNamespace
from retina import Retina, Zone, ZoneRule, IoUTracker
from retina.detect import Detection

class FromSupervision:
    """Wrap any per-frame sv.Detections producer → a Retina detector."""
    def __init__(self):
        self._x = 0
    def __call__(self, frame):
        self._x += 6
        sv_like = SimpleNamespace(            # what an sv.Detections exposes
            xyxy=[[self._x, 40, self._x + 20, 60]],
            confidence=[0.91],
            class_id=[0],
            data={"class_name": ["person"]},
        )
        return Detection.from_supervision(sv_like)   # → list[Detection]

dock = Zone("dock", [(40, 0), (60, 0), (60, 100), (40, 100)])
cam = Retina("cam_01", FromSupervision(), [ZoneRule(dock, classes={"person"})],
             tracker=IoUTracker(min_hits=1))
for event in cam.run(frames):
    print(event.to_json())   # zone.enter / zone.exit on the ingested boxes
```

## 4. Attach a latent `vec`

Retina's state is **dual**: every entity keeps a readable symbolic core *and* an
optional model-tagged latent `vec` on the same record. Attach your own embedding
through the `track.user["vec"]` slot and it flows onto `entity.vec` automatically:

```python
import numpy as np
from retina import Pipeline, DetectorNode, TrackerNode, IoUTracker, Vec, WorldState

pipe = Pipeline([DetectorNode(Walker()), TrackerNode(IoUTracker(min_hits=2))])
ws = None
for i in range(4):
    frame = pipe.process(np.zeros((100, 100, 3), np.uint8), float(i))
    for trk in frame.tracks:
        trk.user["vec"] = Vec(model="my-reid/v0", dim=4, values=[0.1, 0.2, 0.3, 0.4]).to_dict()
    ws = WorldState.from_frame(frame)

print(ws.to_json())
# {"src":"cam","t":3.0,"frame":3,"entities":[{"id":"1","type":"person",...,
#   "vec":{"model":"my-reid/v0","dim":4,"dtype":"fp32","values":[0.1,0.2,0.3,0.4]}}]}
```

For a **real** producer, drop `DinoV2Embedder` (frozen DINOv2, per-object) right
after the tracker — `pip install 'trio-retina[dino]'`:

```python
from retina import DinoV2Embedder, WorldStateNode
pipe = DetectorNode(yolo) | TrackerNode() | DinoV2Embedder() | WorldStateNode()
# each entity.vec now carries a genuine self-supervised DINOv2 embedding
```

## 5. Validate an event stream against the schema

The `retina.event` standard ships a pure-Python validator (and a formal JSON
Schema for other languages). `validate()` returns a list of problems — empty
means valid.

```python
from retina import validate

good = {"type": "zone.enter", "t": 1718254799.8, "src": "cam_01", "id": 42, "label": "person"}
bad  = {"type": "zone.enter", "id": 42}            # missing required t / src

print(validate(good))   # []
print(validate(bad))    # ["missing required field: 't'", "missing required field: 'src'"]
```

Validate a whole stream — e.g. the bundled sample (ships in the wheel, offline):

```python
import json, retina
from retina import validate

problems = 0
for line in open(retina.sample_events()):
    if line.strip() and validate(json.loads(line)):
        problems += 1
print("all valid" if problems == 0 else f"{problems} bad event(s)")
```

## 6. The CLI

The `retina` console script runs the moment `pip install trio-retina` finishes —
the demo is numpy-only (no model, GPU, or video):

```bash
retina demo                          # synthetic dock scene → retina.event stream
retina validate events.jsonl         # check a JSONL stream against retina.event/0.1
retina run workflow.json clip.mp4    # run a declarative pipeline over a source ([video])
retina bench                         # Retina-layer overhead, ms/frame
retina --version
```

Validate the bundled sample with zero setup:

```bash
retina validate "$(python -c 'import retina; print(retina.sample_events())')"
# → 5 event(s): 5 valid, 0 invalid
```

No footage for the video path? Generate a tiny clip (synthetic moving shapes,
cached under `~/.cache/trio-retina/`) and run a workflow over it:

```bash
CLIP="$(python -c 'import retina; print(retina.sample_video())')"   # needs [video]
retina run workflow.json "$CLIP"
```

> The synthetic clip exercises the *video-source plumbing* only — it has no real
> people/vehicles, so a real `YoloDetector` finds nothing in it. For the
> YOLO-on-real-footage path, point Retina at **your own** clip. See the
> [FAQ](faq.md#which-clip-does-sample_video-return).

See the [CLI reference](cli.md) for every flag.
