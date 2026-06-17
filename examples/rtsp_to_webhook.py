"""Camera -> restricted-zone alert -> webhook — the backend-dev path.

Runs out of the box with NO model, NO GPU, NO network: with no args it walks a
scripted "person" into a restricted zone and PRINTS each event. The two lines a
real deployment changes are marked SWAP below.

    python examples/rtsp_to_webhook.py
    python examples/rtsp_to_webhook.py rtsp://CAM https://my-hook   # real camera

Same code shape either way — the rule, zone, and event stream are identical
whether the boxes come from a scripted stub or a live RTSP feed.
"""

import sys

import numpy as np

from retina import Retina, WebhookSink, Zone, ZoneRule
from retina.detect import Detection
from retina.sources import video_frames


class ScriptedDetector:
    """Stub 'model': walks one person left-to-right, one step per frame."""

    def __init__(self):
        self.f = 0

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        x = self.f * 6
        self.f += 1
        return [Detection("person", (x - 10, 40, x + 10, 60), 0.9)] if x <= 96 else []


def main(src: str | None, webhook_url: str | None) -> None:
    # The restricted zone (normalized 0..1 coords, so the same rule works at any
    # resolution). A person whose centroid enters it fires `zone.enter`.
    restricted = Zone("restricted", [(0.4, 0), (0.6, 0), (0.6, 1), (0.4, 1)], normalized=True)

    # SWAP 1 (source): no arg -> scripted frames that run anywhere; a real
    # rtsp:// URL -> pull live frames off the camera with `live=True`.
    if src is None:
        detector = ScriptedDetector()
        frames = [(np.zeros((100, 100, 3), np.uint8), float(i)) for i in range(18)]
    else:
        from retina import YoloDetector  # needs retina-sdk[yolo]

        detector = YoloDetector("yolo11n.pt", classes={"person"})
        frames = video_frames(src, live=True)  # RTSP / HLS / webcam, wall-clock t

    # SWAP 2 (sink): no URL -> just print; a webhook URL -> POST each event as
    # JSON to your backend / queue (stdlib urllib, no `requests` dependency).
    sinks = [WebhookSink(webhook_url)] if webhook_url else []

    cam = Retina(
        source_id="cam_01",
        detector=detector,
        rules=[ZoneRule(restricted, classes={"person"})],
        sinks=sinks,
    )

    for event in cam.run(frames):
        print(event.to_json())


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else None
    webhook_url = sys.argv[2] if len(sys.argv) > 2 else None
    main(src, webhook_url)
