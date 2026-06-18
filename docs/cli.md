# CLI

`pip install trio-retina` puts a `retina` console script on your PATH. It is
stdlib-only on the always-imported path, so `retina demo` runs the instant the
install finishes — no model, no GPU, no video. Anything heavy (OpenCV for a video
source) is lazy-imported by the subcommand that needs it, with a pointer to the
right extra.

```bash
retina --version
```

## `retina demo`

Run the built-in synthetic dock scene (numpy-only) and print the `retina.event`
stream — the fastest way to confirm the install works and see the event format.

```bash
retina demo            # → 5 events: count.threshold, zone.enter, dwell, line.cross, exit
retina demo -q         # suppress the summary line on stderr (stream only)
```

It mirrors `examples/quickstart.py`: a stand-in detector walks one "person" across
a dock zone past a tripwire.

## `retina validate`

Validate a JSONL event stream against `retina.event/0.1`. Prints a valid/invalid
tally and the first problems; **exits non-zero** if any line is invalid, so it
drops straight into CI.

```bash
retina validate events.jsonl
retina validate events.jsonl --max-problems 20

# validate the bundled sample (ships in the wheel, fully offline):
retina validate "$(python -c 'import retina; print(retina.sample_events())')"
# → 5 event(s): 5 valid, 0 invalid
```

## `retina run`

Run a declarative workflow (a `Pipeline.from_json` spec — "n8n without a GUI")
over a video file, `rtsp://` URL, or webcam index. Needs OpenCV for the source
(`pip install 'trio-retina[video]'`); without it you get a clear hint.

```bash
retina run workflow.json clip.mp4              # events to stdout (JSONL)
retina run workflow.json rtsp://cam/stream     # a live source
retina run workflow.json 0                      # webcam index 0
retina run workflow.json clip.mp4 --jsonl out.jsonl   # write to a file instead
```

No footage handy? Generate a tiny synthetic clip (cached under
`~/.cache/trio-retina/`) for exercising the source plumbing:

```bash
CLIP="$(python -c 'import retina; print(retina.sample_video())')"
retina run workflow.json "$CLIP"
```

(The synthetic clip has no real people/vehicles — it verifies the *wiring*, not a
real detector. See the [FAQ](faq.md#which-clip-does-sample_video-return).)

## `retina bench`

Micro-benchmark the **Retina-layer overhead** (tracker + rules + event assembly,
detector excluded) in ms/frame — the honest "how cheap is the core?" number.

```bash
retina bench
retina bench --frames 5000 --tracks 50 --warmup 500
```

The core is numpy and runs on CPU; your real frame budget is set by the *detector*
you put in front, not by Retina. See the [FAQ](faq.md#cpu-or-gpu).
