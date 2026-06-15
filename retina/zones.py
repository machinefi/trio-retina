"""Spatial primitives: zones and lines.

These are *inputs* you define (by hand, from a JSON file, or — in the
commercial layer — auto-discovered by SAM). Retina takes them as plain
geometry; it never depends on how they were authored. Coordinates may be pixels
or normalized 0..1 (set `normalized=True` and pass frame size at match time).
"""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import Point, point_in_polygon, segments_intersect


@dataclass(frozen=True, slots=True)
class Zone:
    """A polygonal region of interest."""

    zone_id: str
    polygon: list[Point]
    normalized: bool = False

    def _scaled(self, size: tuple[int, int] | None) -> list[Point]:
        if not self.normalized:
            return self.polygon
        if size is None:
            raise ValueError("normalized zone requires frame size (w, h) at match time")
        w, h = size
        return [(x * w, y * h) for x, y in self.polygon]

    def contains(self, p: Point, frame_size: tuple[int, int] | None = None) -> bool:
        return point_in_polygon(p, self._scaled(frame_size))


@dataclass(frozen=True, slots=True)
class Line:
    """A directed tripwire a->b. Crossing direction is reported relative to it."""

    line_id: str
    a: Point
    b: Point
    normalized: bool = False

    def _scaled(self, size: tuple[int, int] | None) -> tuple[Point, Point]:
        if not self.normalized:
            return self.a, self.b
        if size is None:
            raise ValueError("normalized line requires frame size (w, h) at match time")
        w, h = size
        return (self.a[0] * w, self.a[1] * h), (self.b[0] * w, self.b[1] * h)

    def crossed(
        self, p_prev: Point, p_now: Point, frame_size: tuple[int, int] | None = None
    ) -> bool:
        a, b = self._scaled(frame_size)
        return segments_intersect(p_prev, p_now, a, b)
