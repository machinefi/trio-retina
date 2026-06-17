# Run trio-retina in your browser

Three zero-install Colab notebooks. Each runs on synthetic detections — no model, no GPU, no network — and ends by printing the standard `retina.event` JSON.

| notebook | what it shows | |
|---|---|---|
| [`01_quickstart_events.ipynb`](01_quickstart_events.ipynb) | detector → tracker → `zone` / `line` / `count` / `dwell` events, then `validate(ev) == []` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machinefi/trio-retina/blob/main/notebooks/01_quickstart_events.ipynb) |
| [`02_camera_to_webhook.ipynb`](02_camera_to_webhook.ipynb) | a restricted-zone alert pushed to a sink (swap in `WebhookSink` + an RTSP camera for production) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machinefi/trio-retina/blob/main/notebooks/02_camera_to_webhook.ipynb) |
| [`03_from_supervision.ipynb`](03_from_supervision.ipynb) | pipe your existing `sv.Detections` straight in via `Detection.from_supervision` (no `supervision` install needed) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machinefi/trio-retina/blob/main/notebooks/03_from_supervision.ipynb) |
