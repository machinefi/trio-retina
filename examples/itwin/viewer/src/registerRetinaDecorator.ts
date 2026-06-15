/*---------------------------------------------------------------------------------------------
 * Drop-in: register the Retina layer in any iTwin Viewer.
 *
 *   import { registerRetinaDecorator } from "./registerRetinaDecorator";
 *   // ...once the iModel view is open:
 *   await registerRetinaDecorator("/retina_events.json");
 *
 * The `placement` below is the ONE per-site calibration: it lands Retina's local
 * ground frame (metres, from the camera→world homography) on this iModel's
 * coordinates. These values are tuned for the Bentley **Baytown** sample iModel
 * (local engineering coords, ground ≈ Z=0, extents X 393.7–421.9 / Y 105.3–140.1)
 * and match the offline preview GIF exactly.
 *-------------------------------------------------------------------------------------------*/

import { IModelApp } from "@itwin/core-frontend";
import { Point3d } from "@itwin/core-geometry";

import { loadRetinaDoc, RetinaDecorator, RetinaEvent } from "./RetinaDecorator";

export async function registerRetinaDecorator(
  jsonUrl = "/retina_events.json",
  onAlert?: (e: RetinaEvent, t: number) => void,
): Promise<RetinaDecorator> {
  const doc = await loadRetinaDoc(jsonUrl);

  const decorator = new RetinaDecorator(doc, {
    origin: Point3d.create(410, 113, 0.1), // Baytown: open slab in front of the vessels
    scale: 0.45, // Retina metres → iModel units
    yaw: (120 * Math.PI) / 180, // align the monitored road across the slab
  });

  // retina.event → twin alerts. Wire this to your notification/UI of choice.
  decorator.onEvents = (events, t) => {
    for (const e of events) onAlert?.(e, t);
  };

  IModelApp.viewManager.addDecorator(decorator);
  decorator.play();
  return decorator;
}
