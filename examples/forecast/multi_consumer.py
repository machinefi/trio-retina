"""One Retina state, three consumers at once.

A single Retina pass over real video produces one standard state stream, and that
SAME stream drives three different consumers at once:

  1. RULES     — geometric events (zone / line / count) off the live tracks
  2. DYNAMICS  — a forecast of the near-future state (LinearForecaster; a learned model next)
  3. LLM-JUDGE — a natural-language read of the state (stub here; swap for a VLM)

An opaque V-JEPA-style latent serves one consumer (its own predictor). One Retina
WorldState serves many — and you can read it. That is what a raw backbone can't
give you, and why Retina is the necessary layer.

    python examples/forecast/multi_consumer.py /tmp/demo.mp4
"""

import os
import sys
from collections import Counter

import cv2

sys.path.insert(0, os.path.dirname(__file__))
from dynamics import LinearForecaster  # noqa: E402

from retina import CountRule, IoUTracker, Line, LineRule, WorldState, YoloDetector, Zone, ZoneRule  # noqa: E402
from retina.nodes import DetectorNode, RuleNode, TrackerNode  # noqa: E402
from retina.pipeline import Pipeline  # noqa: E402
from retina.sources import video_frames  # noqa: E402

CLASSES = {"car", "truck", "bus", "motorcycle"}
FPS, MAX_FRAMES = 5, 120


def llm_judge(ws: WorldState) -> str:
    """Stub for an LLM/VLM reading the state. Deterministic, no API — swap for a
    real model with a key. The point: it consumes the SAME readable state."""
    types = Counter(e.type for e in ws.entities)
    busiest = types.most_common(1)[0][0] if types else "—"
    return f"{sum(types.values())} objects ({dict(types)}); mostly {busiest}"


def main(path):
    cap = cv2.VideoCapture(path)
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    stride = max(1, round(native / FPS))

    road = Zone("road", [(0.1, 0.5), (0.95, 0.5), (0.95, 0.68), (0.1, 0.68)], normalized=True)
    door = Line("mid", (0.45, 0.45), (0.45, 0.7), normalized=True)
    pipe = Pipeline(
        [DetectorNode(YoloDetector("yolo11n.pt", classes=CLASSES, min_confidence=0.3)),
         TrackerNode(IoUTracker(min_hits=3, max_missed=8)),
         RuleNode(ZoneRule(road, classes=CLASSES, dwell_s=2.0)),
         RuleNode(LineRule(door, classes=CLASSES)),
         RuleNode(CountRule(threshold=4, classes=CLASSES, zone=road))],
        source_id="cam",
    )

    print(f"one Retina pass over {os.path.basename(path)} → three consumers\n", file=sys.stderr)
    fc = LinearForecaster()
    events, judge_lines = [], []
    states = []
    for img, t in video_frames(path, stride=stride, max_frames=MAX_FRAMES):
        f = pipe.process(img, t)
        events.extend(f.events)
        ws = WorldState.from_frame(f)
        fc.observe(ws)
        states.append(ws)
        if int(t * FPS) % 25 == 0 and ws.entities:  # sample the LLM consumer
            judge_lines.append(f"  t={t:4.1f}s  {llm_judge(ws)}")

    print("── 1. RULES consumer (events off the live state) ──")
    for e in events[:6]:
        print(f"  {e.to_json()}")
    print(f"  … {len(events)} events total\n")

    print("── 2. DYNAMICS consumer (forecast 0.6s ahead) ──")
    pred = fc.predict(0.6)
    for e in pred.entities[:5]:
        cx, cy = (e.bbox[0] + e.bbox[2]) / 2, (e.bbox[1] + e.bbox[3]) / 2
        print(f"  {e.type} #{e.id} → predicted centroid ({cx:.0f}, {cy:.0f})")
    print()

    print("── 3. LLM-JUDGE consumer (NL read of the same state) ──")
    for line in judge_lines[:5]:
        print(line)
    print("\nSame state, three consumers — swap the backbone (YOLO↔V-JEPA, see any_model.py)\n"
          "and all three keep working. That's why Retina is the layer.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/demo.mp4")
