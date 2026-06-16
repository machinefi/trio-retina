"""Unit tests for CountRule comparators and emit_initial behavior (model-free)."""

import pytest

from retina import CountRule, Track
from retina.events import EventType


def _track(track_id, label="person", x=0):
    return Track(
        track_id=track_id,
        label=label,
        bbox=(x, 0, x + 10, 10),
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
