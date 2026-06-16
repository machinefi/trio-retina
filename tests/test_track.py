"""Unit tests for the dependency-free IoUTracker (re-association & id stability)."""

from retina import IoUTracker
from retina.detect import Detection


def _det(x1, y1, x2, y2, label="person", conf=0.9):
    return Detection(label=label, bbox=(x1, y1, x2, y2), confidence=conf)


def test_confirmation_after_min_hits():
    trk = IoUTracker(iou_threshold=0.3, min_hits=3, max_missed=15)
    box = (10, 10, 30, 30)
    # frames 1,2: not yet confirmed -> nothing surfaced
    assert trk.update([_det(*box)], 0.0) == []
    assert trk.update([_det(*box)], 1.0) == []
    # frame 3: seen >= min_hits -> confirmed, surfaced
    out = trk.update([_det(*box)], 2.0)
    assert len(out) == 1
    assert out[0].confirmed is True


def test_id_stable_across_motion():
    trk = IoUTracker(iou_threshold=0.3, min_hits=2, max_missed=15)
    trk.update([_det(0, 0, 20, 20)], 0.0)
    out = trk.update([_det(5, 0, 25, 20)], 1.0)  # overlapping move -> same track
    assert len(out) == 1
    tid = out[0].track_id
    out2 = trk.update([_det(9, 0, 29, 20)], 2.0)  # keep overlapping
    assert len(out2) == 1
    assert out2[0].track_id == tid  # id preserved across motion


def test_reassociation_after_short_occlusion_keeps_id():
    trk = IoUTracker(iou_threshold=0.3, min_hits=2, max_missed=15)
    box = (10, 10, 30, 30)
    trk.update([_det(*box)], 0.0)
    out = trk.update([_det(*box)], 1.0)  # confirmed
    tid = out[0].track_id
    # occluded for a few frames (< max_missed): no detection
    assert trk.update([], 2.0) == []
    assert trk.update([], 3.0) == []
    # re-detected near its last box -> same id re-associates
    out2 = trk.update([_det(11, 11, 31, 31)], 4.0)
    assert len(out2) == 1
    assert out2[0].track_id == tid


def test_track_dropped_after_max_missed():
    trk = IoUTracker(iou_threshold=0.3, min_hits=1, max_missed=2)
    box = (10, 10, 30, 30)
    out = trk.update([_det(*box)], 0.0)  # min_hits=1 -> confirmed immediately
    tid = out[0].track_id
    # miss it for more than max_missed frames -> dropped from internal list
    trk.update([], 1.0)
    trk.update([], 2.0)
    trk.update([], 3.0)  # missed=3 > max_missed=2 -> dropped
    # a re-detection now gets a NEW id (old track is gone)
    out2 = trk.update([_det(*box)], 4.0)
    assert len(out2) == 1
    assert out2[0].track_id != tid


def test_new_object_gets_new_id():
    trk = IoUTracker(iou_threshold=0.3, min_hits=1, max_missed=15)
    out1 = trk.update([_det(0, 0, 20, 20)], 0.0)
    id1 = out1[0].track_id
    # a far-away, non-overlapping detection is a different object -> new id
    out2 = trk.update([_det(0, 0, 20, 20), _det(200, 200, 220, 220)], 1.0)
    ids = sorted(t.track_id for t in out2)
    assert id1 in ids
    assert len(ids) == 2
    assert ids[1] != id1


def test_two_objects_keep_distinct_stable_ids():
    trk = IoUTracker(iou_threshold=0.3, min_hits=2, max_missed=15)
    a, b = (0, 0, 20, 20), (200, 200, 220, 220)
    trk.update([_det(*a), _det(*b)], 0.0)
    out = trk.update([_det(*a), _det(*b)], 1.0)
    by_pos = {t.bbox[0]: t.track_id for t in out}
    id_a, id_b = by_pos[0], by_pos[200]
    assert id_a != id_b
    # move both a bit, overlapping their own previous boxes
    out2 = trk.update([_det(5, 0, 25, 20), _det(205, 200, 225, 220)], 2.0)
    by_pos2 = {round(t.bbox[0]): t.track_id for t in out2}
    assert by_pos2[5] == id_a
    assert by_pos2[205] == id_b
