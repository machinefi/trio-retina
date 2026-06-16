"""Unit tests for the pure geometry helpers (no ML, no deps)."""

from retina.geometry import (
    iou,
    point_in_polygon,
    segments_intersect,
    side_of_line,
)


# --- iou ---


def test_iou_disjoint_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_identical_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_half_overlap_exact():
    # Two 10x10 boxes overlapping in a 5x10 strip.
    # inter = 50, union = 100 + 100 - 50 = 150 -> 1/3.
    a = (0, 0, 10, 10)
    b = (5, 0, 15, 10)
    assert iou(a, b) == 50.0 / 150.0


def test_iou_containment():
    # Inner 5x5 fully inside outer 10x10: inter=25, union=100 -> 0.25.
    outer = (0, 0, 10, 10)
    inner = (0, 0, 5, 5)
    assert iou(outer, inner) == 0.25


# --- point_in_polygon ---


_SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_point_in_polygon_inside():
    assert point_in_polygon((5.0, 5.0), _SQUARE) is True


def test_point_in_polygon_outside():
    assert point_in_polygon((20.0, 5.0), _SQUARE) is False
    assert point_in_polygon((-1.0, 5.0), _SQUARE) is False


def test_point_in_polygon_half_open_shared_edge_counted_once():
    # Two squares sharing the vertical edge at x=10. A point exactly on that
    # shared edge must be counted by EXACTLY ONE of the two zones (half-open).
    left = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    right = [(10.0, 0.0), (20.0, 0.0), (20.0, 10.0), (10.0, 10.0)]
    p = (10.0, 5.0)
    in_left = point_in_polygon(p, left)
    in_right = point_in_polygon(p, right)
    assert in_left != in_right  # exactly one, never both, never neither


# --- side_of_line ---


def test_side_of_line_sign():
    a = (0.0, 0.0)
    b = (0.0, 10.0)  # directed straight up (in image coords, +y is down)
    # cross product: (b-a) x (p-a). For a vertical a->b line, a point to the
    # left/right gives opposite signs; a point on the line gives 0.
    left = side_of_line((-5.0, 5.0), a, b)
    right = side_of_line((5.0, 5.0), a, b)
    on = side_of_line((0.0, 5.0), a, b)
    assert left > 0
    assert right < 0
    assert on == 0.0


# --- segments_intersect ---


def test_segments_intersect_crossing():
    # An X: the two diagonals cross at the center.
    assert segments_intersect((0, 0), (10, 10), (0, 10), (10, 0)) is True


def test_segments_intersect_non_crossing():
    # Parallel, offset — never meet.
    assert segments_intersect((0, 0), (10, 0), (0, 5), (10, 5)) is False
    # Disjoint, non-parallel — don't reach each other.
    assert segments_intersect((0, 0), (1, 1), (5, 0), (6, 1)) is False


def test_segments_intersect_endpoint_exactly_on_line_counts():
    # p1->p2 ends exactly on the tripwire a->b: sign 0 differs from the side
    # of p1, so it must count as a crossing (path sampled onto the wire).
    wire_a = (5.0, 0.0)
    wire_b = (5.0, 10.0)  # vertical line at x=5
    p1 = (0.0, 5.0)
    p2 = (5.0, 5.0)  # lands exactly on the wire
    assert segments_intersect(p1, p2, wire_a, wire_b) is True


def test_segments_intersect_collinear_along_does_not_fire():
    # A path lying ALONG the line (both endpoints collinear with a->b) must NOT
    # fire — only a genuine crossing does.
    wire_a = (0.0, 0.0)
    wire_b = (10.0, 0.0)  # horizontal line y=0
    p1 = (2.0, 0.0)
    p2 = (8.0, 0.0)  # both on the line, moving along it
    assert segments_intersect(p1, p2, wire_a, wire_b) is False
