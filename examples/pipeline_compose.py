"""Compose models into a pipeline — the "n8n without a GUI" demo.

Wire nodes (detector -> tracker -> VLM/V-JEPA enricher -> rule) with the `|`
operator. Add, remove, or reorder a node without touching the others — the same
idea as cortex's YOLO+gate+VLM chain, but composable. No model, no GPU.

    python examples/pipeline_compose.py
"""

import numpy as np

from retina import CallableDetector, EnricherNode, IoUTracker, Zone, ZoneRule
from retina.detect import Detection


class Walker:
    def __init__(self):
        self.f = 0

    def __call__(self, image):
        x = self.f * 6
        self.f += 1
        return [Detection("person", (x - 10, 40, x + 10, 60), 0.9)] if x <= 96 else []


def vlm_note(frame):
    """Stand-in for a real VLM / V-JEPA enricher: attach context to the frame.
    Swap this function body for a Qwen-VL / Gemini call returning a dict."""
    if frame.tracks:
        return {"scene": f"{len(frame.tracks)} person(s) near dock"}


dock = Zone("dock", [(0.4, 0), (0.6, 0), (0.6, 1), (0.4, 1)], normalized=True)

# The whole pipeline, LCEL-style. Each step is swappable behind a tiny protocol.
pipe = (
    CallableDetector(Walker())          # detector  (swap for YoloDetector / a VLM)
    | IoUTracker(min_hits=2)            # tracker   (swap for Norfair / ByteTrack)
    | EnricherNode(vlm_note)            # enricher  (where a VLM / V-JEPA node slots in)
    | ZoneRule(dock, classes={"person"}, dwell_s=2.0)  # rule -> events
)


def main():
    print("pipeline:", "  |  ".join(type(n).__name__ for n in pipe.nodes), "\n")
    for i in range(14):
        f = pipe.process(np.zeros((100, 100, 3), np.uint8), float(i))
        for ev in f.events:
            print(f"  t={i:>2}  {ev.type:<12} scene={f.user.get('scene')!r}")
    print("\nSame pipeline as a shareable JSON workflow -> examples/workflow.json")
    print("  Pipeline.from_json('examples/workflow.json').run(video_frames('v.mp4'))")


if __name__ == "__main__":
    main()
