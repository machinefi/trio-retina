"""Unit tests for event sinks (JSONL round-trip)."""

import json

from retina import Event, JsonlSink


def test_jsonl_sink_writes_one_round_trippable_line_per_event(tmp_path):
    path = tmp_path / "events.jsonl"
    sink = JsonlSink(str(path))
    events = [
        Event(type="zone.enter", t=1.0, src="cam", id=1, label="person"),
        Event(type="line.cross", t=2.0, src="cam", id=2, dir="a_to_b"),
        Event(type="count.threshold", t=3.0, src="cam", n=5),
    ]
    for e in events:
        sink(e)
    sink.close()

    lines = path.read_text().splitlines()
    assert len(lines) == len(events)  # one JSON line per event
    for line, ev in zip(lines, events, strict=True):
        back = json.loads(line)  # each line round-trips
        assert back == ev.to_dict()
