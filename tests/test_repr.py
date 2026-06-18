"""Concise `__repr__` on the core data types: salient fields present, noise gone."""

from retina import Detection, Entity, Event, EventType, Line, Relation, Track, Vec, WorldState, Zone
from retina.events import Frame


def test_event_repr_is_concise():
    r = repr(Event(type=EventType.ZONE_ENTER, t=1718254799.8, src="cam", id=42, label="person", zone="dock"))
    assert r.startswith("Event(zone.enter")
    assert "id=42" in r and "label='person'" in r and "zone='dock'" in r and "t=1718254799.8" in r
    # No noise from the many None optional fields.
    assert "None" not in r
    assert "conf=" not in r and "dir=" not in r and "box=" not in r


def test_detection_repr():
    r = repr(Detection(label="person", bbox=(32, 40, 52, 60), confidence=0.90))
    assert r == "Detection(label='person' bbox=(32,40,52,60) conf=0.90)"


def test_track_repr():
    r = repr(Track(track_id=7, label="person", bbox=(1, 2, 3, 4), confidence=0.9,
                   first_seen=1.0, last_seen=4.0, confirmed=True))
    assert "id=7" in r and "label='person'" in r and "dwell=3s" in r
    assert "None" not in r and "prev_centroid" not in r and "user" not in r


def test_worldstate_repr():
    ws = WorldState(src="cam_01", t=3.0, entities=[Entity(id=str(i), type="person") for i in range(14)])
    assert repr(ws) == "WorldState(src='cam_01' t=3.0 entities=14)"


def test_vec_repr():
    assert repr(Vec(model="dinov2-small", dim=384)) == "Vec(model='dinov2-small' dim=384)"


def test_entity_repr():
    r = repr(Entity(id="7", type="person", conf=0.91))
    assert "id='7'" in r and "type='person'" in r and "conf=0.91" in r
    assert "None" not in r and "attrs" not in r


def test_relation_repr():
    assert repr(Relation("a", "b", "near")) == "Relation('a' -near-> 'b')"


def test_zone_and_line_repr():
    assert repr(Zone("dock", [(0, 0), (1, 0), (1, 1), (0, 1)], normalized=True)) == \
        "Zone(id='dock' pts=4 normalized)"
    assert "Line(id='door'" in repr(Line("door", (0, 0), (1, 1)))


def test_frame_repr_counts_not_contents():
    f = Frame(frame_num=5, src="cam", t=2.0)
    f.detections = [1, 2, 3]
    r = repr(f)
    assert "#5" in r and "dets=3" in r and "tracks=0" in r and "events=0" in r
    assert "image" not in r
