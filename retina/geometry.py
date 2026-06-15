"""Pure geometry helpers — no ML, no deps beyond stdlib.

Pixel-space math shared by tracking (IoU association) and event rules
(point-in-zone, line-crossing). Coordinates are (x, y) with origin top-left,
bboxes are (x1, y1, x2, y2).
"""

from __future__ import annotations

Point = tuple[float, float]
BBox = tuple[float, float, float, float]


def centroid(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def point_in_polygon(p: Point, polygon: list[Point]) -> bool:
    """Ray-casting test. `polygon` is an ordered ring of >=3 vertices.

    Boundary is *half-open* (left/top edges inside, right/bottom outside) — the
    standard convention so adjacent zones tile without double-counting a point on
    a shared edge."""
    x, y = p
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def side_of_line(p: Point, a: Point, b: Point) -> float:
    """Signed side of point `p` relative to directed line a->b.
    >0 left, <0 right, ~0 on the line (2D cross product)."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


_EPS = 1e-9


def _sign(v: float) -> int:
    if v > _EPS:
        return 1
    if v < -_EPS:
        return -1
    return 0


def segments_intersect(p1: Point, p2: Point, a: Point, b: Point) -> bool:
    """Do segment p1->p2 and segment a->b cross?

    Orientation-based: each segment's endpoints must fall on opposite sides of
    the other. A point landing *exactly* on the line counts as a crossing (sign
    0 differs from ±1), so a path sampled onto the tripwire isn't missed — but a
    path lying *along* the line (both endpoints collinear) does not fire."""
    d1 = _sign(side_of_line(p1, a, b))
    d2 = _sign(side_of_line(p2, a, b))
    d3 = _sign(side_of_line(a, p1, p2))
    d4 = _sign(side_of_line(b, p1, p2))
    return d1 != d2 and d3 != d4
