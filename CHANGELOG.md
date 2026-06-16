# Changelog

All notable changes to Retina are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Top-OSS documentation pass: tagline, table of contents, Features list, and
  reorganized README sections.
- `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`, and GitHub issue / pull
  request templates.

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

[Unreleased]: https://github.com/machinefi/trio-retina/compare/main...HEAD
[0.0.3]: https://github.com/machinefi/trio-retina
[0.0.2]: https://github.com/machinefi/trio-retina
[0.0.1]: https://github.com/machinefi/trio-retina
