"""Unit tests for the model-free event rules: CountRule comparators /
emit_initial, the ZoneRule exit-grace window, and anchor (feet/head/center)."""

import pytest

from retina import CountRule, Track, Zone, ZoneRule
from retina.events import EventType


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
