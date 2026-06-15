"""Real-model demo: YOLO on a video file -> events to JSONL.

    pip install 'retina-sdk[all]'
    python examples/yolo_video.py path/to/video.mp4

Same code shape as quickstart — only the detector and source change. That's the
point: the rules, tracker, events, and sinks are identical whether the boxes
come from a scripted stub, YOLO, or a VLM.
"""

import sys

from retina import CountRule, JsonlSink, Line, LineRule, Retina, Zone, ZoneRule, YoloDetector
from retina.sources import video_frames


def main(path: str) -> None:
    # Normalized coords (0..1) so the same rules work at any resolution.
    dock = Zone("dock", [(0.3, 0.2), (0.7, 0.2), (0.7, 0.9), (0.3, 0.9)], normalized=True)
    door = Line("door", (0.5, 0.0), (0.5, 1.0), normalized=True)

    cam = Retina(
        source_id="cam_01",
        detector=YoloDetector("yolo11n.pt", classes={"person"}),
        rules=[
            ZoneRule(dock, classes={"person"}, dwell_s=10.0),
            LineRule(door, classes={"person"}),
            CountRule(threshold=3, classes={"person"}, zone=dock),
        ],
        sinks=[JsonlSink("events.jsonl")],
    )

    # Normalized zones self-scale: Retina backfills the frame size on the first
    # frame, so you never compute pixel coordinates by hand.
    n = 0
    for event in cam.run(video_frames(path, stride=2)):
        print(event.to_json())
        n += 1
    print(f"\n{n} events -> events.jsonl", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python examples/yolo_video.py <video.mp4>", file=sys.stderr)
        raise SystemExit(2)
    main(sys.argv[1])
