"""Tests for the assembled `WorldState` snapshot and its `from_frame` builder."""

import json

from retina import Entity, Frame, Relation, Track, Vec, WorldState


def _track(track_id, label, bbox, conf=0.9, t=1.0, **user):
    return Track(
        track_id=track_id,
        label=label,
        bbox=bbox,
        confidence=conf,
        first_seen=t,
        last_seen=t,
        confirmed=True,
        user=user,
    )


def test_to_dict_omits_empty_and_carries_vec():
    ws = WorldState(
        src="cam_01",
        t=1718000000.0,
        frame=42,
        entities=[
            Entity(
                id="7",
                type="person",
                bbox=(10.0, 20.0, 60.0, 180.0),
                conf=0.91,
                vec=Vec(model="osnet-reid", dim=512, values=[0.1, 0.2]),
            ),
            Entity(id="9", type="forklift", bbox=(200.0, 40.0, 320.0, 210.0)),
        ],
        relations=[Relation(subj="7", obj="9", predicate="near")],
        scene=Vec(model="v-jepa2-vitl", dim=1024, dtype="fp16", ref="vec://abc123"),
    )
    d = ws.to_dict()

    # required keys present, empty `user` omitted
    assert d["src"] == "cam_01" and d["t"] == 1718000000.0 and d["frame"] == 42
    assert "user" not in d

    # entity with no extras stays minimal (no conf/attrs/vec keys)
    forklift = d["entities"][1]
    assert forklift == {"id": "9", "type": "forklift", "bbox": [200.0, 40.0, 320.0, 210.0]}

    # vec carries its model tag and dim, ref/values omitted appropriately
    person_vec = d["entities"][0]["vec"]
    assert person_vec["model"] == "osnet-reid" and person_vec["dim"] == 512
    assert person_vec["values"] == [0.1, 0.2] and "ref" not in person_vec

    scene = d["scene"]
    assert scene["model"] == "v-jepa2-vitl" and scene["dim"] == 1024
    assert scene["ref"] == "vec://abc123" and "values" not in scene

    # relation minimal: no family/conf
    assert d["relations"] == [{"subj": "7", "obj": "9", "predicate": "near"}]


def test_minimal_worldstate_is_two_keys():
    assert WorldState(src="c", t=0.0).to_dict() == {"src": "c", "t": 0.0}


def test_entity_locus_emitted_only_when_set():
    # Default: no locus key at all (omit-empty, like bbox/conf/vec).
    bare = Entity(id="1", type="thing").to_dict()
    assert "locus" not in bare

    # Set: emitted as a plain list, distinct from bbox.
    e = Entity(id="subject", type="rf_subject", locus=(2.41, 3.07))
    d = e.to_dict()
    assert d == {"id": "subject", "type": "rf_subject", "locus": [2.41, 3.07]}
    assert "bbox" not in d  # locus is not a bbox

    # bbox (pixels) and locus (metric world frame) can coexist independently.
    both = Entity(
        id="2", type="person", bbox=(10.0, 20.0, 60.0, 180.0), locus=(1.0, 2.0, 0.5)
    ).to_dict()
    assert both["bbox"] == [10.0, 20.0, 60.0, 180.0]
    assert both["locus"] == [1.0, 2.0, 0.5]


def test_entity_repr_notes_locus_presence():
    assert "locus" not in repr(Entity(id="1", type="thing"))
    assert "locus" in repr(Entity(id="1", type="thing", locus=(0.0, 0.0)))


def test_entity_locus_round_trips_through_json():
    ws = WorldState(
        src="csi_room",
        t=1.0,
        entities=[
            Entity(
                id="subject",
                type="rf_subject",
                locus=(2.41, 3.07),
                vec=Vec(model="csi-jepa/v0", dim=2, values=[0.1, 0.2]),
            )
        ],
        scene=Vec(model="csi-jepa/v0", dim=2, values=[0.3, 0.4]),
    )
    back = json.loads(ws.to_json())
    assert back == ws.to_dict()
    assert back["entities"][0]["locus"] == [2.41, 3.07]


def test_old_worldstate_without_locus_still_parses():
    # An additive optional field must not change pre-0.2 payloads: a state with no
    # locus serializes exactly as before (no spurious locus key anywhere).
    ws = WorldState(
        src="cam_01",
        t=1.0,
        entities=[Entity(id="7", type="person", bbox=(10.0, 20.0, 60.0, 180.0))],
    )
    d = ws.to_dict()
    assert d["entities"][0] == {"id": "7", "type": "person", "bbox": [10.0, 20.0, 60.0, 180.0]}


def test_json_round_trips():
    ws = WorldState(
        src="cam_01",
        t=1.0,
        entities=[Entity(id="1", type="person", vec=Vec(model="facenet", dim=128))],
        relations=[Relation(subj="1", obj="2", predicate="holds", family="functional", conf=0.8)],
        scene=Vec(model="v-jepa2", dim=1024, ref="vec://x"),
    )
    back = json.loads(ws.to_json())
    assert back == ws.to_dict()
    assert back["entities"][0]["vec"] == {"model": "facenet", "dim": 128, "dtype": "fp32"}
    assert back["relations"][0]["family"] == "functional" and back["relations"][0]["conf"] == 0.8


def test_from_frame_builds_entities_from_tracks():
    frame = Frame(
        frame_num=12,
        src="cam_02",
        t=99.5,
        tracks=[
            _track(7, "person", (10.0, 20.0, 60.0, 180.0), conf=0.88),
            _track(
                9,
                "forklift",
                (200.0, 40.0, 320.0, 210.0),
                conf=0.75,
                vec={"model": "osnet-reid", "dim": 512, "values": [0.3, 0.4]},
            ),
        ],
    )
    ws = WorldState.from_frame(frame)

    assert ws.src == "cam_02" and ws.t == 99.5 and ws.frame == 12
    assert len(ws.entities) == 2 and not ws.relations and ws.scene is None

    person = ws.entities[0]
    assert person.id == "7" and person.type == "person"
    assert person.bbox == (10.0, 20.0, 60.0, 180.0) and person.conf == 0.88
    assert person.vec is None

    forklift = ws.entities[1]
    assert isinstance(forklift.vec, Vec)
    assert forklift.vec.model == "osnet-reid" and forklift.vec.dim == 512

    # assembled snapshot serializes through the same omit-empty path
    d = ws.to_dict()
    assert d["entities"][1]["vec"]["values"] == [0.3, 0.4]
    assert "relations" not in d and "scene" not in d


def test_from_frame_lifts_scene_latent_from_frame_user():
    # A scene-level latent in frame.user["scene"] (as a dict) lifts onto
    # ws.scene — symmetric with how per-track vec lifts onto entity.vec.
    frame = Frame(
        frame_num=5,
        src="cam_03",
        t=12.0,
        tracks=[_track(1, "person", (0.0, 0.0, 10.0, 20.0))],
        user={"scene": {"model": "vjepa2:vitl", "dim": 1024, "values": [0.5, 0.6, 0.7]}},
    )
    ws = WorldState.from_frame(frame)

    assert isinstance(ws.scene, Vec)
    assert ws.scene.model == "vjepa2:vitl" and ws.scene.dim == 1024
    assert ws.scene.values == [0.5, 0.6, 0.7]
    # and it round-trips through the omit-empty serializer
    assert ws.to_dict()["scene"]["values"] == [0.5, 0.6, 0.7]


def test_from_frame_without_scene_leaves_scene_none():
    frame = Frame(frame_num=1, src="c", t=0.0, tracks=[])
    assert WorldState.from_frame(frame).scene is None


def test_from_frame_ignores_unknown_vec_keys():
    # A foreign/future producer adding an extra key must not crash the frame:
    # unknown keys are dropped, the known fields still build a valid Vec.
    frame = Frame(
        frame_num=1,
        src="c",
        t=0.0,
        tracks=[_track(1, "person", (0.0, 0.0, 10.0, 20.0),
                       vec={"model": "m", "dim": 4, "junk": 1})],
    )
    ws = WorldState.from_frame(frame)
    vec = ws.entities[0].vec
    assert isinstance(vec, Vec) and vec.model == "m" and vec.dim == 4


def test_from_frame_malformed_vec_degrades_to_none():
    # Missing the required model/dim -> degrade to vec=None rather than crash.
    frame = Frame(
        frame_num=1,
        src="c",
        t=0.0,
        tracks=[_track(1, "person", (0.0, 0.0, 10.0, 20.0), vec={"values": [0.1]})],
    )
    ws = WorldState.from_frame(frame)
    assert ws.entities[0].vec is None


def test_from_frame_malformed_scene_degrades_to_none():
    frame = Frame(
        frame_num=1, src="c", t=0.0,
        tracks=[_track(1, "person", (0.0, 0.0, 10.0, 20.0))],
        user={"scene": {"dim": 8}},  # missing model -> None, no crash
    )
    assert WorldState.from_frame(frame).scene is None
