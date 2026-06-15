# The iTwin Viewer side

`RetinaDecorator.ts` is a plain [iTwin.js](https://www.itwinjs.org/) `Decorator`
— it drops into any iTwin Viewer and depends only on `@itwin/core-frontend`
(+ `@itwin/core-geometry`, `@itwin/core-common`). It replays
[`../retina_events.json`](../retina_events.json) as live markers + forecast
arrows + event alerts on the iModel.

## Files

| File | Role |
|------|------|
| [`src/RetinaDecorator.ts`](./src/RetinaDecorator.ts) | The integration itself — markers, forecast arrows, the monitored zone, playback. |
| [`src/registerRetinaDecorator.ts`](./src/registerRetinaDecorator.ts) | One call to load the JSON, calibrate placement, and add the decorator to a view. |

## Wire it into a viewer

Modern iTwin.js (**5.x**) installs natively on Apple Silicon and opens a **local
snapshot iModel with no cloud auth**. Scaffold a viewer (web or desktop), then:

```ts
import { registerRetinaDecorator } from "./registerRetinaDecorator";

// after the iModelConnection + ScreenViewport are open:
await registerRetinaDecorator("/retina_events.json", (e, t) => {
  // surface retina.event as a twin alert in your UI
  myAlertPanel.push(`${e.type} #${e.id ?? "?"} ${e.zone ?? ""} @ ${t}s`);
});
```

That's the whole integration. Swap the iModel, the camera, or the detector
upstream — this code doesn't change.

## Run against the Baytown sample iModel

The preview was built on Bentley's **Baytown** sample (a process plant). To run
the *interactive* viewer:

1. Get a snapshot `.bim`. Baytown ships in
   [`imodeljs/desktop-starter`](https://github.com/imodeljs/desktop-starter)
   under `assets/Baytown.bim` (≈22 MB, opens with no auth via `SnapshotDb` /
   `SnapshotConnection`).
2. Scaffold an iTwin Viewer (e.g. `@itwin/web-viewer-react` or
   `@itwin/desktop-viewer-react`, iTwin.js **5.x**) pointed at that snapshot.
3. Serve `retina_events.json` (copy it into the app's `public/`) and call
   `registerRetinaDecorator()` once the view opens.

The `Placement` in `registerRetinaDecorator.ts` (`origin (410,113,0.1)`,
`scale 0.45`, `yaw 120°`) is calibrated to Baytown's extents and matches the GIF.
Point it at a different iModel by re-running the one-time calibration.

## Note on the preview GIF (honest scope)

The README GIF was rendered **headless, with no GPU**: the iTwin.js **backend**
(`@itwin/core-backend`) exported the real Baytown geometry
(`exportGraphics` → 461k triangles), a tiny software rasterizer produced the plant
image, and [`../render/overlay_twin.py`](../render/overlay_twin.py) drew the same
Retina layer this decorator draws — so the preview is reproducible anywhere. The
**interactive** decorator here is the real thing and needs a WebGL viewer (a GPU
box). Same data, same placement, same overlay logic.
