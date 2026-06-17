# Changelog

All notable changes to Trio Retina are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- World-model hero visual: `media/world_model_hero.gif`, an honest looping
  imagination rollout rendered from the real trained dynamics model on a held-out
  synthetic sequence (magenta = imagined, gray = actual), plus a minimal
  `media/world_model_seam.png` (any encoder → one WorldState → any dynamics). The
  README first screen now leads with the world-model story. Generators:
  `examples/world_model/make_hero_gif.py` and `make_seam_png.py`.
- `retina.embed.DinoV2Embedder`: the first real producer for the latent `vec`
  channel — a frozen DINOv2 per-object embedder (sizes small/base/large, dim
  384/768/1024) that crops each track and attaches a genuine self-supervised
  embedding flowing into `WorldState`/`entity.vec`. Lazy torch import (numpy-only
  core); install via `pip install 'trio-retina[dino]'`. See
  `examples/dino_embeddings.py`.
- `retina.embed.VJepa2Embedder`: the first real producer for the scene latent —
  a frozen V-JEPA 2 *video* encoder that pools a rolling `clip_len`-frame clip
  into one `frame.user["scene"]` → `ws.scene`. Lazy torch import (numpy-only
  core); install via `pip install 'trio-retina[vjepa]'`.
- `WorldState.from_frame` now lifts a scene latent from `frame.user["scene"]`
  (a `{model,dim,…}` dict) onto `ws.scene` — symmetric with the per-track `vec`.
- `examples/world_model/multi_encoder.py`: "swap the encoder, the state schema
  is constant" demo — the same pipeline run symbolic-only, with `DinoV2Embedder`
  (per-object `entity.vec`), and with `VJepa2Embedder` (scene-level `ws.scene`).
- `examples/world_model/{dataset,dynamics_model,dynamics}.py`: a Dreamer-4-style
  latent-dynamics back-end — a small transformer world model trained offline on
  recorded `WorldState` sequences, with autoregressive imagination rollouts. The
  honest appearance ablation (held-out, real DINOv2 vecs) shows the latent
  channel improves multi-step motion prediction: pos+appearance < pos-only <
  constant-velocity. Torch lives only in the examples (numpy-only core); install
  via the new `[dynamics]` extra. Numpy-only smoke test in
  `tests/test_world_model.py`.
- `examples/world_model/end_to_end.py`: the full world-model stack in one
  runnable script — perception encoder → Retina `WorldState` (symbolic + DINOv2
  latent) → learned dynamics → imagination rollout, printing a frame's
  `WorldState` and the imagined-vs-truth trajectory. Reuses the Phase-2 dataset
  and model code.
- `examples/world_model/benchmark.py` + `BENCHMARK.md`: a small front/back-end
  benchmark grid over {const-velocity, pos-only, pos+appearance} × prediction
  horizon, writing a held-out position-error table. Captured REAL on Mac Studio
  (MPS, real DINOv2): at horizon 7, pos+appearance 1.33 px beats const-velocity
  7.68 px (+83%) and pos-only 1.45 px (+8%); the appearance edge widens with the
  horizon. Framed as early/illustrative (synthetic scene, small PoC).

## [0.2.1] — 2026-06-17

### Added

- `examples/bench_overhead.py`: honest numpy-only micro-benchmark of the Retina-layer
  overhead (tracker + rules + event build, detector excluded) in ms/frame.
- `tests/test_sources.py`: unit tests for the live-source path using a fake capture
  stub — read-failure→recovery and slow-consumer drop-to-latest run with no cv2 and
  no real RTSP (via the `capture_factory` injection seam).
- Colab notebooks (`notebooks/`): runnable, zero-install quickstart, camera→webhook,
  and from-Supervision demos that print `retina.event` JSON on synthetic input.

### Changed

- `video_frames` no longer ends the generator on a transient live (`rtsp://` /
  `live=True` / webcam) `cap.read()` failure: it reconnects with exponential
  backoff and, for live sources, drops to the latest frame under back-pressure.
  Finite files are unchanged — a real EOF still ends the generator and every
  frame is delivered (no reconnect, no dropping).
- `CountRule(threshold)` now accepts `threshold` positionally, so `CountRule(3)`
  works; `CountRule(threshold=3)` is unchanged.
- README headline quickstart now runs on a bare `pip install trio-retina` (numpy
  only, no model / video) via a stand-in detector, with the YOLO + `video_frames`
  form moved to a clearly-labeled `[yolo]` block below it.

### Fixed

- `CountRule(3)` no longer raises a confusing `TypeError` from the keyword-only
  `threshold` (front-door friction for new users).

## [0.2.0] — 2026-06-17

### Added

- Example `examples/latent_vec.py`: populate the dual-state latent `vec` channel by
  hand (attach your own embedding → `WorldState` entity → serialize → round-trip),
  runnable with numpy only — shows the latent interface is usable before the
  built-in producers ship.
- `Detection.from_supervision(detections, class_names=None)`: ingest a Roboflow
  Supervision `sv.Detections` into `list[Detection]` by duck-typing (no
  `supervision` import), so Supervision users plug straight into Retina's event
  layer. Labels resolve from `data["class_name"]`, then a `class_names` mapping,
  then `str(class_id)`; missing `confidence` / `class_id` are handled.
- `LineRule(min_frames=...)`: a crossing is confirmed only after the track stays
  on the new side for `min_frames` frames (default 1 = unchanged instant-emit),
  suppressing single-frame jitter near the tripwire — mirrors Supervision's
  `LineZone.minimum_crossing_threshold`.
- Examples: `examples/rtsp_to_webhook.py` (camera → restricted-zone alert →
  webhook) and `examples/from_supervision.py` (ingest a Roboflow `sv.Detections`
  pipeline), both runnable with no model / GPU / network on synthetic input.

### Changed

- `[all]` extra now installs every optional adapter's deps, including `grounding`
  (transformers + torch + pillow), so it is genuinely "everything" as the README says.
- Honesty pass on forecast claims: dropped the unreproducible "−35%" learned-vs-baseline
  number from the README hero caption and the forecast README (the training footage
  isn't redistributed). `quick_forecast.py` is documented as the reproducible,
  no-footage demo; the real-video result is clearly marked as needing your own clip.
- README latent-channel wording reconciled with `DESIGN.md`: the latent `vec` is a
  shipped, serializable *interface*; the automatic V-JEPA / ReID *producers* are
  roadmap, not shipped.
- Noted that examples ship with the source tree (`git clone`), not the installed wheel.

## [0.1.0] — 2026-06-17

First public open-source release.

### Added

- `ZoneRule(exit_grace_s=...)`: a track stays logically inside until it has been
  out-of-zone or absent for `exit_grace_s` seconds, so a single-frame detection
  blip or id flicker no longer emits a spurious `zone.exit` or resets the dwell
  timer. The exit `dur` is measured to the last frame seen inside.
- `anchor` param on `ZoneRule` / `CountRule` (`center` default, `feet`, `head`)
  selecting which body-point of the bbox tests zone membership.

## [0.0.4]

### Added

- Rebranded to **Trio Retina**; repository moved to the `machinefi` org.
- Top-OSS documentation pass (tagline, Features, reorganized README) plus
  `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue / pull-request templates.
- `WorldStateNode` + `Pipeline.run_states()`: the assembled-state channel now
  flows through the composable pipeline (and a `worldstate` workflow node).
- `Pipeline.process(frame_num=...)` to set the frame index explicitly.
- Retina × iTwin.js digital-twin example (`examples/itwin/`) — the event/state
  stream rendered live on a Bentley iModel.
- MkDocs Material documentation site + GitHub Pages deploy workflow; `CITATION.cff`;
  Dependabot for pip + GitHub Actions.
- Tests: 28 → 70, covering the geometry primitives, `MotionGate`, tracker
  re-association, `CountRule` comparators, vec validation, and the new APIs.

### Changed

- `IoUTracker` association is vectorized with numpy (≈5–20× on the matching step
  in crowded scenes); zone/line geometry is scaled once per frame, not per object.
- `Event.vec` accepts a `Vec` or a dict and normalizes on serialize, unifying the
  latent representation across the event and world-state channels.
- PyPI metadata: search-led description, expanded keywords / classifiers, full
  project URLs.

### Fixed / Security

- `validate()` now checks the `vec` sub-object (parity with `event.schema.json`).
- `LearnedForecaster` loads checkpoints with `weights_only=True` (was an unsafe
  pickle load); the iTwin overlay drops `subprocess(shell=True)`; the iTwin
  decorator builds tooltips with `textContent`, not `innerHTML` (XSS).

### Removed

- The AutoResearch / auto-tune examples and the unimplemented `TDMPC2Dynamics`
  stub, plus stale terminology across docs and example docstrings.

## [0.0.3]

### Added

- `WorldState`: the assembled entity + relation + latent state snapshot
  (`Entity`, `Relation`, `Vec`).
- Forecast demo (`examples/forecast/`): the dynamics layer (L4) on top of Retina,
  including a real-video baseline, a one-state / three-consumers demo, a learned
  dynamics model that beats a constant-velocity baseline, and the annotated demo
  GIF behind the README.

## [0.0.2]

### Added

- Packaging metadata, CI across Python 3.10–3.13, contributing guide, and badges.
- `DESIGN.md`: Retina in the world-model stack and the L1 two-axis evolution.

## [0.0.1]

### Added

- Initial release of Retina — the model-agnostic state layer for world models.
- The `retina.event` standard ([`SPEC.md`](SPEC.md)) with a JSON Schema and a
  pure-Python validator.
- Composable pipeline (`|` operator, explicit node list, and JSON workflow files).
- Detectors (`YoloDetector`, `GroundingDinoDetector`, `VlmDetector`,
  `CallableDetector`), trackers (`IoUTracker`, `NorfairTracker`), rules
  (`ZoneRule`, `LineRule`, `CountRule`), gates (`MotionGate`), and sinks
  (`JsonlSink`, `WebhookSink`).
- `event_f1` / `match_events`: a generic metric to compare two event streams.

[Unreleased]: https://github.com/machinefi/trio-retina/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/machinefi/trio-retina/releases/tag/v0.2.1
[0.2.0]: https://github.com/machinefi/trio-retina/releases/tag/v0.2.0
[0.1.0]: https://github.com/machinefi/trio-retina/releases/tag/v0.1.0
[0.0.4]: https://github.com/machinefi/trio-retina
[0.0.3]: https://github.com/machinefi/trio-retina
[0.0.2]: https://github.com/machinefi/trio-retina
[0.0.1]: https://github.com/machinefi/trio-retina
