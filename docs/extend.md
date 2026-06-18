# Extend Retina

Every stage is a tiny `Protocol` — implement one and it drops straight into a
pipeline and composes with `|`. The core never imports a model, so "add your own
X" is the normal path, not an escape hatch. For dev setup (tests, lint), see
[`CONTRIBUTING.md`](https://github.com/machinefi/trio-retina/blob/main/CONTRIBUTING.md).

## Add your own detector

A **detector** is anything callable `frame -> list[Detection]`. Wrap a plain
function with `CallableDetector` (which also gives you class / confidence
filtering and `|` composition for free):

```python
import numpy as np
from retina import CallableDetector, IoUTracker, ZoneRule, Zone
from retina.detect import Detection

def my_model(frame: np.ndarray) -> list[Detection]:
    # call YOUR model here; return one Detection per object found
    return [Detection(label="person", bbox=(10, 10, 30, 30), confidence=0.8)]

detector = CallableDetector(my_model, classes={"person"}, min_confidence=0.5)
pipe = detector | IoUTracker() | ZoneRule(Zone("z", [(0, 0), (40, 0), (40, 40), (0, 40)]))
```

A class satisfies the `Detector` protocol just as well — implement
`__call__(self, frame) -> list[Detection]`. (Already on Roboflow Supervision?
`Detection.from_supervision(sv_dets)` ingests an `sv.Detections` — see
[cookbook recipe 3](cookbook.md#3-ingest-an-existing-supervision-svdetections).)

## Add your own tracker

A **tracker** implements `update(detections, t) -> list[Track]`. Swap in
ByteTrack / OC-SORT / BoT-SORT behind this one method; Retina only needs that
each returned `Track` carries a stable `track_id` (and `prev_centroid` for
line-crossing). The built-in `IoUTracker` is the dependency-free default;
`NorfairTracker` wraps Norfair.

```python
from retina.track import Track

class MyTracker:
    def __init__(self):
        self._next = 0
    def update(self, detections, t):
        out = []
        for d in detections:                 # toy: a fresh id per detection
            self._next += 1
            out.append(Track(track_id=self._next, label=d.label, bbox=d.bbox,
                             confidence=d.confidence, first_seen=t, last_seen=t,
                             confirmed=True))
        return out
```

Use it via `TrackerNode(MyTracker())` (or `Retina(..., tracker=MyTracker())`).

## Add your own rule

A **rule** is a small stateful machine: `update(tracks, t, frame_idx) -> list[Event]`.
Emit `Event`s on transitions, using the closed `retina.event` vocabulary so your
output stays standard. Subclass `EventRule` for the `|` mixin and normalized-coord
support:

```python
from retina.rules import EventRule
from retina.events import Event

class FirstPersonRule(EventRule):
    """Emit one count.threshold the first frame any person appears."""
    def __init__(self):
        self._fired = False
    def update(self, tracks, t, frame_idx):
        if self._fired:
            return []
        people = [trk for trk in tracks if trk.label == "person"]
        if not people:
            return []
        self._fired = True
        return [Event(type="count.threshold", t=t, src="", n=len(people),
                      frame=frame_idx, ext={"threshold": 1, "cmp": ">="})]
```

Leave `src=""` and the `RuleNode` stamps it with the frame's source for you. Keep
rules **model-free and deterministic** — semantic / LLM judgment belongs one layer
up, not in a rule.

## Add your own sink

A **sink** is `__call__(event) -> None`. The shipped `JsonlSink` / `WebhookSink`
are a few lines each; a Kafka / MQTT / DB sink follows the same shape:

```python
class PrintSink:
    def __call__(self, event):
        print(event.to_json())

from retina import Retina
cam = Retina("cam_01", detector, rules=[...], sinks=[PrintSink()])
```

Wrap it as `SinkNode(PrintSink())` to place it explicitly in a `|` chain.

## Gates and enrichers

- A **gate** is `callable(image, t) -> bool` — return `False` to skip the detector
  on a boring frame (the cascade pattern that cuts model cost). `MotionGate` ships;
  use yours via `GateNode(my_gate)` or `Retina(..., gate=my_gate)`.
- An **enricher** is `callable(frame) -> dict | value` whose result is merged into
  `frame.user` — the seam for a VLM caption, a classifier, or a latent producer.
  Wire it with `EnricherNode(my_fn)`.

## Register a node type for declarative workflows

To make your step usable from a `Pipeline.from_json(...)` workflow, register a
builder that maps a JSON node spec to a `Node`:

```python
from retina import register_node, SinkNode

register_node("print", lambda spec: SinkNode(PrintSink()))
# now {"type": "print", "id": "out"} is valid in a workflow.json
```

→ The protocols live in `retina/detect.py`, `retina/track.py`, `retina/rules.py`,
`retina/export.py`, and `retina/nodes.py`. Keep additions small and composable —
see the design principles in
[`CONTRIBUTING.md`](https://github.com/machinefi/trio-retina/blob/main/CONTRIBUTING.md).
