/*---------------------------------------------------------------------------------------------
 * Retina × iTwin.js — the integration seam.
 *
 * A single iTwin.js `Decorator` that replays a Retina event/state stream
 * (`retina_events.json`, the `retina.event` standard) on top of ANY iModel:
 *
 *     camera ─▶ Retina (any backbone) ─▶ retina.event + WorldState ─▶ THIS decorator ─▶ iTwin twin
 *
 * It is deliberately model-agnostic and backend-free: it never imports a CV
 * model, never touches pixels, and depends only on `@itwin/core-frontend`. A
 * Bentley engineer drops this file into an iTwin Viewer, points it at the JSON,
 * and the twin gains a live, semantic, *predictive* layer:
 *   - each detected entity becomes a marker on the iModel ground plane,
 *   - the forecast (next ~1 s) is drawn as a world arrow,
 *   - Retina events (zone enter / line cross / count) surface as alerts.
 *
 * Coordinates: each entity carries a `world` ground-plane point (metres) that
 * Retina produced via a one-time camera→world homography. `placement` maps that
 * local metric frame onto this particular iModel's spatial extents (origin +
 * scale + optional yaw) — the only thing you tune per site.
 *-------------------------------------------------------------------------------------------*/

import {
  DecorateContext,
  Decorator,
  GraphicType,
  IModelApp,
  Marker,
} from "@itwin/core-frontend";
import { Point3d, Vector3d } from "@itwin/core-geometry";
import { ColorByName, ColorDef, LinePixels } from "@itwin/core-common";

// ---- the wire format (a subset of retina.event / WorldState) --------------------------------

export interface RetinaEntity {
  id: string;
  type: string;
  img: [number, number];
  world: [number, number]; // ground-plane metres (camera→world homography)
  zone?: string;
  forecast?: { world: [number, number]; horizon_s: number };
}

export interface RetinaEvent {
  type: string;
  t: number;
  src: string;
  id?: number;
  zone?: string;
  label?: string;
  [k: string]: unknown;
}

export interface RetinaFrame {
  t: number;
  entities: RetinaEntity[];
  events: RetinaEvent[];
}

export interface RetinaDoc {
  meta: {
    schema: string;
    source: string;
    fps: number;
    image_size: [number, number];
    forecaster: string;
    world: { units: string; road_rect_m: [number, number][]; [k: string]: unknown };
    [k: string]: unknown;
  };
  frames: RetinaFrame[];
}

/** Maps Retina's local metric ground frame onto this iModel's coordinates. */
export interface Placement {
  origin: Point3d; // where Retina (0,0) lands in the iModel
  scale?: number; // metres → iModel units (default 1)
  yaw?: number; // rotation about Z, radians (align road with the model)
}

// ---- per-type colour, so two "teams"/classes read at a glance -------------------------------

const TYPE_COLOR: Record<string, number> = {
  car: ColorByName.cyan,
  truck: ColorByName.orange,
  bus: ColorByName.yellow,
  motorcycle: ColorByName.magenta,
  person: ColorByName.springGreen,
};
const colorFor = (type: string) =>
  ColorDef.create(TYPE_COLOR[type] ?? ColorByName.white);

// ---- entity marker: a labelled disc with a live tooltip -------------------------------------

class EntityMarker extends Marker {
  constructor(world: Point3d, ent: RetinaEntity) {
    super(world, { x: 18, y: 18 });
    const c = colorFor(ent.type);
    this.title = makeTooltip(ent);
    this.setScaleFactor({ low: 0.6, high: 1.4 }); // shrink with distance
    this.drawFunc = (ctx: CanvasRenderingContext2D) => {
      ctx.beginPath();
      ctx.arc(0, 0, 8, 0, Math.PI * 2);
      ctx.fillStyle = c.toRgbaString(0.85);
      ctx.strokeStyle = ent.zone ? "white" : c.toRgbaString(1);
      ctx.lineWidth = ent.zone ? 2.5 : 1.5; // ring entities currently in a zone
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "white";
      ctx.font = "11px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(`#${ent.id}`, 0, -12);
    };
  }
}

function makeTooltip(ent: RetinaEntity): HTMLElement {
  // textContent / DOM nodes, never innerHTML — the event stream is untrusted
  // (a crafted type/zone string must not be able to inject HTML into the viewer).
  const div = document.createElement("div");
  const head = document.createElement("b");
  head.textContent = `${ent.type} #${ent.id}`;
  div.appendChild(head);
  if (ent.zone) {
    div.appendChild(document.createElement("br"));
    div.appendChild(document.createTextNode(`zone: ${ent.zone}`));
  }
  if (ent.forecast) {
    div.appendChild(document.createElement("br"));
    const coords = ent.forecast.world.map((v) => v.toFixed(1)).join(", ");
    div.appendChild(document.createTextNode(`forecast +${ent.forecast.horizon_s}s → (${coords}) m`));
  }
  return div;
}

// ---- the decorator ---------------------------------------------------------------------------

export class RetinaDecorator implements Decorator {
  private frameIndex = 0;
  private timer?: ReturnType<typeof setInterval>;
  public onEvents?: (events: RetinaEvent[], t: number) => void;

  constructor(private doc: RetinaDoc, private placement: Placement) {}

  /** Resolve a Retina ground point (metres) into iModel world coordinates. */
  private toWorld([x, y]: [number, number]): Point3d {
    const s = this.placement.scale ?? 1;
    const yaw = this.placement.yaw ?? 0;
    const cx = Math.cos(yaw),
      sx = Math.sin(yaw);
    const rx = x * cx - y * sx;
    const ry = x * sx + y * cx;
    const o = this.placement.origin;
    return Point3d.create(o.x + rx * s, o.y + ry * s, o.z);
  }

  // --- playback ---
  public play(): void {
    if (this.timer) return;
    const dt = 1000 / (this.doc.meta.fps || 5);
    this.timer = setInterval(() => this.step(), dt);
  }
  public pause(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = undefined;
  }
  public setFrame(i: number): void {
    this.frameIndex = Math.max(0, Math.min(i, this.doc.frames.length - 1));
    this.invalidate();
  }
  private step(): void {
    this.frameIndex = (this.frameIndex + 1) % this.doc.frames.length;
    const f = this.doc.frames[this.frameIndex];
    if (f.events.length && this.onEvents) this.onEvents(f.events, f.t);
    this.invalidate();
  }
  private invalidate(): void {
    IModelApp.viewManager.invalidateDecorationsAllViews();
  }

  // --- the actual drawing, called by iTwin every redraw ---
  public decorate(context: DecorateContext): void {
    this.drawZone(context);
    const frame = this.doc.frames[this.frameIndex];
    if (!frame) return;
    for (const ent of frame.entities) {
      const here = this.toWorld(ent.world);
      new EntityMarker(here, ent).addDecoration(context);
      if (ent.forecast) this.drawArrow(context, here, this.toWorld(ent.forecast.world), colorFor(ent.type));
    }
  }

  /** The calibrated road zone, drawn as a translucent ground rectangle. */
  private drawZone(context: DecorateContext): void {
    const rect = (this.doc.meta.world.road_rect_m ?? []).map((p) => this.toWorld(p as [number, number]));
    if (rect.length < 3) return;
    const b = context.createGraphicBuilder(GraphicType.WorldDecoration);
    b.setSymbology(ColorDef.create(ColorByName.green), ColorDef.create(ColorByName.green).withAlpha(40), 2);
    b.addShape([...rect, rect[0]]);
    context.addDecorationFromBuilder(b);
  }

  /** Forecast vector: a world-overlay arrow from the entity to its predicted point. */
  private drawArrow(context: DecorateContext, from: Point3d, to: Point3d, color: ColorDef): void {
    const b = context.createGraphicBuilder(GraphicType.WorldOverlay);
    b.setSymbology(color, color, 4, LinePixels.Solid);
    b.addLineString([from, to]);
    // arrowhead
    const dir = Vector3d.createStartEnd(from, to);
    const len = dir.magnitude();
    if (len > 1e-3) {
      dir.scaleInPlace(1 / len);
      const back = to.plusScaled(dir, -Math.min(1.5, len * 0.3));
      const perp = Vector3d.create(-dir.y, dir.x, 0).scale(Math.min(1.0, len * 0.2));
      b.addLineString([back.plus(perp), to, back.minus(perp)]);
    }
    context.addDecorationFromBuilder(b);
  }
}

/** Fetch + parse a retina_events.json document. */
export async function loadRetinaDoc(url: string): Promise<RetinaDoc> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to load ${url}: ${res.status}`);
  return (await res.json()) as RetinaDoc;
}
