# Event spec

The `retina.event` interchange format — a standard, model-agnostic, app-agnostic
format for *what happened* in a stream. Designed to be boring and tiny, like a
JWT: the smallest valid event is three fields (`type`, `t`, `src`).

```json
{"type": "line.cross", "t": 1718254799.8, "src": "cam_01"}
```

The full specification — required and optional fields, the closed set of
primitive event types, the latent (`vec`) channel and `WorldState`, serialization
(JSON Lines), validation, and versioning — is the canonical
[**`SPEC.md`**](https://github.com/machinefi/trio-retina/blob/main/SPEC.md) in the
repository. It is kept there as the single source of truth so this site never
drifts from the standard.
