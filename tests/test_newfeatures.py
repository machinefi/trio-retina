"""Unit tests for the newer features: WorldState in the pipeline, Vec on Event,
explicit frame_num."""

import numpy as np

from retina import (
    CallableDetector,
    Event,
    IoUTracker,
    Pipeline,
    Vec,
    WorldState,
    WorldStateNode,
)
from retina.detect import Detection


class _ScriptedDetector:
    """Emits one steady 'person' box on every frame."""

    def __call__(self, image):
        return [Detection(label="person", bbox=(40, 40, 60, 60), confidence=0.9)]


def _pipe_with_worldstate():
    return Pipeline(
        [
            CallableDetector(_ScriptedDetector()),
            IoUTracker(min_hits=1),  # confirm immediately so a track surfaces
            WorldStateNode(),
        ]
    )


def test_worldstate_node_attaches_worldstate_with_one_entity_per_track():
    pipe = _pipe_with_worldstate()
    f = pipe.process(np.zeros((100, 100, 3), np.uint8), 0.0)
    ws = f.user["worldstate"]
    assert isinstance(ws, WorldState)
    assert len(ws.entities) == len(f.tracks) == 1
    ent = ws.entities[0]
    assert ent.type == "person"
    assert ent.id == str(f.tracks[0].track_id)


def test_run_states_yields_worldstate_per_frame_matching_tracks():
    pipe = _pipe_with_worldstate()
    frames = [(np.zeros((100, 100, 3), np.uint8), float(i)) for i in range(4)]
    states = list(pipe.run_states(frames))
    assert len(states) == len(frames)
    assert all(isinstance(s, WorldState) for s in states)
    # one steady person -> one entity in every state once confirmed
    for s in states:
        assert len(s.entities) == 1
        assert s.entities[0].type == "person"


def test_event_vec_accepts_vec_object_and_serializes_to_dict():
    v = Vec(model="v-jepa2-vitl", dim=1024, dtype="fp16", ref="vec://x")
    e = Event(type="zone.enter", t=1.0, src="cam", id=7, vec=v)
    d = e.to_dict()
    assert isinstance(d["vec"], dict)
    assert d["vec"]["model"] == "v-jepa2-vitl"
    assert d["vec"]["dim"] == 1024
    assert d["vec"]["ref"] == "vec://x"


def test_pipeline_process_explicit_frame_num():
    pipe = _pipe_with_worldstate()
    f = pipe.process(np.zeros((100, 100, 3), np.uint8), 0.0, frame_num=123)
    assert f.frame_num == 123
    # the assembled worldstate carries the explicit frame number too
    assert f.user["worldstate"].frame == 123


def test_pipeline_process_default_frame_num_is_monotonic():
    pipe = _pipe_with_worldstate()
    img = np.zeros((100, 100, 3), np.uint8)
    f0 = pipe.process(img, 0.0)
    f1 = pipe.process(img, 1.0)
    assert f0.frame_num == 0
    assert f1.frame_num == 1
