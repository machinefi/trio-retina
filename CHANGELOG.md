# Changelog

All notable changes to Trio Retina are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/machinefi/trio-retina/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/machinefi/trio-retina/releases/tag/v0.1.0
[0.0.4]: https://github.com/machinefi/trio-retina
[0.0.3]: https://github.com/machinefi/trio-retina
[0.0.2]: https://github.com/machinefi/trio-retina
[0.0.1]: https://github.com/machinefi/trio-retina
