# FAQ & troubleshooting

Short answers to the questions that come up first. See also the
[concepts](concepts.md) and the [cookbook](cookbook.md).

## Which extra do I install?

The core is **numpy-only** — `pip install trio-retina` is enough to build
pipelines, emit events, validate, and run the CLI demo. Heavy runtimes live behind
extras and are imported lazily, so importing `retina` never pulls in torch:

| You want… | Install | Pulls in |
|---|---|---|
| events on synthetic / your own detections | `trio-retina` | numpy |
| a real **YOLO** detector | `'trio-retina[yolo]'` | ultralytics |
| read **video / RTSP / webcam** files | `'trio-retina[video]'` | opencv-python |
| **open-vocab** from text (Grounding DINO) | `'trio-retina[grounding]'` | transformers + torch |
| per-object **DINOv2** `vec` producer | `'trio-retina[dino]'` | torch + transformers |
| scene-level **V-JEPA 2** producer | `'trio-retina[vjepa]'` | torch + transformers |
| everything | `'trio-retina[all]'` | all of the above |

If you instantiate an adapter without its extra, you get a friendly error naming
the right one (e.g. `YoloDetector needs ultralytics. Install with: pip install 'trio-retina[yolo]'`).

## I get no events — why?

Walk down the chain:

- **No tracker, but a tracked rule.** `LineRule` and `CountRule` (and a meaningful
  `ZoneRule` dwell) need *identity* over time. `Retina(...)` adds an `IoUTracker`
  by default; if you build a raw `Pipeline`, include a `TrackerNode()`.
- **`min_hits` too high for a short clip.** `IoUTracker(min_hits=2)` won't confirm
  a track that's only seen once. Lower it (or feed more frames) while debugging.
- **A `classes={...}` filter that doesn't match your labels.** The label is
  whatever your detector emits (`"person"`, `"0"`, `"car"`, …). Print a few
  detections first.
- **Edge-triggered rules already satisfied.** `CountRule` fires on the
  *transition* into the threshold, not every frame it stays true; pass
  `emit_initial=True` to fire on frame 1 if it's already over.
- **A gate dropping frames.** A `MotionGate` skips static frames — expected on a
  frozen test image after the first.

Quick sanity check: `retina demo` should always print five events. If it does, the
install is fine and the issue is in your rules/wiring.

## RTSP keeps dropping / reconnecting

`video_frames(src, live=True)` (or any `rtsp://` / webcam-index source) is hardened
for unattended use: a transient `read()` failure is treated as a drop, not EOF, and
the capture is **re-opened with exponential backoff** (`reconnect_initial` →
`reconnect_max`), bounded by `max_reconnect_attempts` / `reconnect_timeout`. A
slow consumer gets **drop-to-latest** back-pressure (only the newest frame is
kept), so latency stays bounded instead of buffering stale frames. A real *file*
is never reconnected — EOF ends its generator exactly as before. See
[`examples/rtsp_to_webhook.py`](https://github.com/machinefi/trio-retina/blob/main/examples/rtsp_to_webhook.py).

## CPU or GPU?

**Retina's core is CPU** — the tracker, rules, geometry, and event assembly are
numpy and cheap (run `retina bench` for ms/frame). The **detector defines your
frame budget**: a YOLO/DINO/VLM model is where the GPU helps, and that's the layer
you choose. So you can run the whole event layer at the edge on CPU and put only
the model on an accelerator if you have one. The latent producers (`DinoV2Embedder`,
`VJepa2Embedder`) auto-select `mps → cuda → cpu`.

## How do I validate events?

In Python, `validate(event_or_dict)` returns a list of problems (empty = valid);
`is_valid(...)` is the boolean form. From the shell, `retina validate stream.jsonl`
checks a whole JSONL file against `retina.event/0.1` and exits non-zero if any line
is invalid. A formal JSON Schema (for other languages / tooling) ships in the wheel
and is returned by `load_schema()`. See [cookbook recipe 5](cookbook.md#5-validate-an-event-stream-against-the-schema).

## Which clip does `sample_video()` return?

A **synthetically generated** clip — deterministic moving shapes on a dark
background, written once with OpenCV and cached under `~/.cache/trio-retina/`. It
exists so the **video-source plumbing** (`video_frames`, `retina run`, striding,
EOF) runs out of the box with zero network and zero third-party-footage licensing
risk. It is *not* real-world footage, so a real `YoloDetector` finds no
people/vehicles in it. For the YOLO-on-real-footage path, point Retina at **your
own** clip. `sample_video()` needs the `[video]` extra to write the clip and
raises a clear hint if OpenCV is missing.

`sample_events()`, by contrast, is a small `retina.event` JSONL **bundled in the
wheel** — fully offline, zero risk, ready for `validate` and the CLI.

## Where are the examples?

In the **repository**, not the installed wheel — `git clone` and run
`python examples/quickstart.py`. The top-level examples run with no model and no
GPU (synthetic detections); real-footage / world-model demos need the extras and a
clip. The [notebooks](https://github.com/machinefi/trio-retina/tree/main/notebooks)
open in Colab with no local install.
