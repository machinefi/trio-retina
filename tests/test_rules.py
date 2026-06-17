"""Unit tests for the model-free event rules: CountRule comparators /
emit_initial, the ZoneRule exit-grace window, and anchor (feet/head/center)."""

import pytest

from retina import CountRule, LineRule, Track, Zone, ZoneRule
from retina.events import EventType
from retina.zones import Line


def _track(track_id, label="person", x=0, bbox=None):
    return Track(
        track_id=track_id,
        label=label,
        bbox=bbox if bbox is not None else (x, 0, x + 10, 10),
        confidence=0.9,
        first_seen=0.0,
        last_seen=0.0,
        confirmed=True,
    )


def _tracks(n):
    return [_track(i) for i in range(n)]


def test_count_rule_rejects_bad_comparator():
    with pytest.raises(ValueError):
        CountRule(threshold=1, comparator="!=")


def test_count_rule_greater_than():
    rule = CountRule(threshold=2, comparator=">")
    assert rule.update(_tracks(2), 0.0, 0) == []  # 2 > 2 is False
    out = rule.update(_tracks(3), 1.0, 1)  # 3 > 2 -> fire
    assert len(out) == 1
    assert out[0].type == EventType.COUNT_THRESHOLD
    assert out[0].n == 3
    assert out[0].ext["cmp"] == ">"


def test_count_rule_less_equal():
    rule = CountRule(threshold=1, comparator="<=")
    # baseline frame establishes prev without firing on a real transition; start
    # above threshold so the predicate is False, then drop to <= to trigger.
    assert rule.update(_tracks(5), 0.0, 0) == []  # 5 <= 1 False
    out = rule.update(_tracks(1), 1.0, 1)  # 1 <= 1 -> fire
    assert len(out) == 1
    assert out[0].n == 1


def test_count_rule_strictly_less():
    rule = CountRule(threshold=2, comparator="<")
    assert rule.update(_tracks(2), 0.0, 0) == []  # 2 < 2 False
    out = rule.update(_tracks(1), 1.0, 1)  # 1 < 2 -> fire
    assert len(out) == 1


def test_count_rule_emit_initial_fires_on_first_frame():
    rule = CountRule(threshold=1, comparator=">=", emit_initial=True)
    out = rule.update(_tracks(3), 0.0, 0)  # already true on frame 1 -> fires
    assert len(out) == 1
    assert out[0].n == 3


def test_count_rule_default_does_not_fire_on_first_frame():
    rule = CountRule(threshold=1, comparator=">=")  # emit_initial=False default
    out = rule.update(_tracks(3), 0.0, 0)  # already true, but baseline -> no fire
    assert out == []


def test_count_rule_edge_triggered_rearm():
    rule = CountRule(threshold=2, comparator=">=")
    assert rule.update(_tracks(0), 0.0, 0) == []  # baseline False
    assert len(rule.update(_tracks(2), 1.0, 1)) == 1  # False -> True fires
    assert rule.update(_tracks(2), 2.0, 2) == []  # stays True, no re-fire
    assert rule.update(_tracks(0), 3.0, 3) == []  # back to False, re-arm
    assert len(rule.update(_tracks(2), 4.0, 4)) == 1  # fires again


# --- Zone anchor: which body-point tests polygon membership -----------------

# Horizontal band 40 <= y <= 60; used so feet/head/center fall on different sides.
_BAND = Zone("band", [(0, 40), (100, 40), (100, 60), (0, 60)])


def test_zone_anchor_rejects_bad_value():
    with pytest.raises(ValueError):
        ZoneRule(_BAND, anchor="elbow")


def test_count_anchor_rejects_bad_value():
    with pytest.raises(ValueError):
        CountRule(threshold=1, anchor="elbow")


def test_zone_anchor_center_default():
    # center=(50,50) inside the band; feet/head fall outside.
    rule = ZoneRule(_BAND)  # default anchor="center"
    out = rule.update([_track(1, bbox=(40, 0, 60, 100))], 0.0, 0)
    assert [e.type for e in out] == [EventType.ZONE_ENTER]


def test_zone_anchor_feet():
    trk = _track(1, bbox=(40, 0, 60, 50))  # center=(50,25) out, feet=(50,50) in
    assert ZoneRule(_BAND).update([trk], 0.0, 0) == []  # center: not inside
    out = ZoneRule(_BAND, anchor="feet").update([trk], 0.0, 0)
    assert [e.type for e in out] == [EventType.ZONE_ENTER]


def test_zone_anchor_head():
    trk = _track(1, bbox=(40, 50, 60, 100))  # center=(50,75) out, head=(50,50) in
    assert ZoneRule(_BAND).update([trk], 0.0, 0) == []
    out = ZoneRule(_BAND, anchor="head").update([trk], 0.0, 0)
    assert [e.type for e in out] == [EventType.ZONE_ENTER]


def test_count_anchor_feet_changes_membership():
    trk = _track(1, bbox=(40, 0, 60, 50))  # feet in band, center out
    assert CountRule(threshold=1, zone=_BAND).update([trk], 0.0, 0) == []
    out = CountRule(threshold=1, zone=_BAND, anchor="feet", emit_initial=True).update(
        [trk], 0.0, 0
    )
    assert len(out) == 1 and out[0].n == 1


# --- Zone exit-grace window -------------------------------------------------

# Simple square zone 0..100; a track at x~50 is inside, far away is outside.
_BOX = Zone("box", [(0, 0), (100, 0), (100, 100), (0, 100)])


def _inside_trk(tid=1):
    return _track(tid, bbox=(40, 40, 60, 60))  # centroid (50,50) inside


def _outside_trk(tid=1):
    return _track(tid, bbox=(200, 200, 220, 220))  # centroid (210,210) outside


def test_grace_zero_immediate_exit_matches_old_behavior():
    rule = ZoneRule(_BOX)  # exit_grace_s default 0.0
    assert [e.type for e in rule.update([_inside_trk()], 0.0, 0)] == [EventType.ZONE_ENTER]
    out = rule.update([_outside_trk()], 1.0, 1)
    assert [e.type for e in out] == [EventType.ZONE_EXIT]
    assert out[0].dur == 1.0  # measured enter(0.0) -> last-inside(0.0)


def test_grace_zero_immediate_exit_on_vanish():
    rule = ZoneRule(_BOX)
    rule.update([_inside_trk()], 0.0, 0)
    out = rule.update([], 1.0, 1)  # track vanished while inside
    assert [e.type for e in out] == [EventType.ZONE_EXIT]
    assert out[0].dur == 1.0
    assert out[0].ext["reason"] == "track_lost"


def test_grace_blip_out_keeps_dwell_no_spurious_exit():
    rule = ZoneRule(_BOX, exit_grace_s=2.0, dwell_s=3.0)
    assert [e.type for e in rule.update([_inside_trk()], 0.0, 0)] == [EventType.ZONE_ENTER]
    assert rule.update([_outside_trk()], 1.0, 1) == []  # 1s out < 2s grace: no exit
    assert rule.update([], 1.5, 2) == []  # vanish within grace: still no exit
    # Back inside at t=3.0; dwell timer never reset, so (3.0-0.0)>=3.0 fires dwell,
    # and no new enter is emitted.
    out = rule.update([_inside_trk()], 3.0, 3)
    assert [e.type for e in out] == [EventType.ZONE_DWELL]
    assert out[0].dur == 3.0


def test_grace_elapsed_fires_one_exit_with_correct_dur():
    rule = ZoneRule(_BOX, exit_grace_s=2.0)
    rule.update([_inside_trk()], 0.0, 0)  # enter, last-inside = 0.0
    rule.update([_inside_trk()], 1.0, 1)  # still inside, last-inside = 1.0
    assert rule.update([_outside_trk()], 2.0, 2) == []  # out 1s < grace
    out = rule.update([_outside_trk()], 3.5, 3)  # out 2.5s >= grace -> exit
    assert [e.type for e in out] == [EventType.ZONE_EXIT]
    assert out[0].dur == 1.0  # enter(0.0) -> last-inside(1.0), not current t
    # Exactly one exit: the state is now gone, a further frame is silent.
    assert rule.update([_outside_trk()], 5.0, 4) == []


def test_grace_reentry_within_window_no_new_enter():
    rule = ZoneRule(_BOX, exit_grace_s=5.0)
    assert len(rule.update([_inside_trk()], 0.0, 0)) == 1  # enter
    assert rule.update([_outside_trk()], 1.0, 1) == []  # within grace
    assert rule.update([_inside_trk()], 2.0, 2) == []  # back inside, no new enter


# --- LineRule: crossing + min_frames jitter debounce ------------------------

# Vertical tripwire at x=50; a track at cx<50 is on one side, cx>50 the other.
_WIRE = Line("wire", (50, 0), (50, 100))


def _line_trk(cx, prev_cx, tid=1):
    """A track whose centroid is at (cx, 50), having moved from (prev_cx, 50)."""
    trk = _track(tid, bbox=(cx - 5, 45, cx + 5, 55))
    trk.prev_centroid = (float(prev_cx), 50.0)
    return trk


def test_linerule_min_frames_invalid():
    with pytest.raises(ValueError):
        LineRule(_WIRE, min_frames=0)


def test_linerule_default_emits_on_intersection():
    rule = LineRule(_WIRE)  # min_frames=1 default: legacy behavior
    out = rule.update([_line_trk(60, 40)], 1.0, 0)  # 40 -> 60 crosses x=50
    assert [e.type for e in out] == [EventType.LINE_CROSS]
    assert out[0].dir == "a_to_b"  # moved to the >0-side -> a_to_b per impl


def test_linerule_default_no_cross_when_no_intersection():
    rule = LineRule(_WIRE)
    assert rule.update([_line_trk(45, 40)], 1.0, 0) == []  # 40 -> 45, no crossing


def test_linerule_min_frames_confirms_after_hold():
    rule = LineRule(_WIRE, min_frames=3)
    # Frame 0: crosses 40 -> 60. Pending, not yet emitted.
    assert rule.update([_line_trk(60, 40)], 0.0, 0) == []
    # Frame 1: stays on the new side (60 -> 65), held=2, still pending.
    assert rule.update([_line_trk(65, 60)], 1.0, 1) == []
    # Frame 2: still on new side (65 -> 70), held=3 == min_frames -> confirm.
    out = rule.update([_line_trk(70, 65)], 2.0, 2)
    assert [e.type for e in out] == [EventType.LINE_CROSS]
    assert out[0].dir == "a_to_b"
    assert out[0].t == 2.0 and out[0].frame == 2  # emitted at confirm frame
    # Frame 3: no further events.
    assert rule.update([_line_trk(75, 70)], 3.0, 3) == []


def test_linerule_min_frames_bounce_back_suppressed():
    rule = LineRule(_WIRE, min_frames=3)
    assert rule.update([_line_trk(60, 40)], 0.0, 0) == []  # cross, pending on >0 side
    # Jitter: centroid drifts back to the original side without a confirmed hold
    # (45 -> 40, no fresh intersection) -> pending discarded, nothing emitted.
    assert rule.update([_line_trk(40, 45)], 1.0, 1) == []
    # Staying on the original side keeps emitting nothing.
    assert rule.update([_line_trk(35, 40)], 2.0, 2) == []
    assert rule.update([_line_trk(30, 35)], 3.0, 3) == []
