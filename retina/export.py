"""Event sinks: get events out of Retina and into your world.

JSONL for files/replay, webhook for pushing to your backend/queue. Sinks are
trivially small on purpose — Kafka/MQTT/DB sinks are a few lines following the
same `__call__(event)` shape.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from .compose import Pipeable
from .events import Event


class _SinkPipeable(Pipeable):
    def to_node(self):
        from .nodes import SinkNode

        return SinkNode(self)


class EventSink(Protocol):
    def __call__(self, event: Event) -> None: ...


def to_jsonl(events: Iterable[Event], path: str) -> int:
    """Write events to a JSONL file. Returns the count written."""
    n = 0
    with open(path, "w") as f:
        for ev in events:
            f.write(ev.to_json() + "\n")
            n += 1
    return n


class JsonlSink(_SinkPipeable):
    """Append events to a JSONL file as they arrive (streaming)."""

    def __init__(self, path: str):
        self._f = open(path, "a")

    def __call__(self, event: Event) -> None:
        self._f.write(event.to_json() + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


class WebhookSink(_SinkPipeable):
    """POST each event as JSON to a URL. Uses urllib (stdlib) — no requests dep."""

    def __init__(self, url: str, *, timeout: float = 5.0):
        self._url = url
        self._timeout = timeout

    def __call__(self, event: Event) -> None:
        import urllib.request

        req = urllib.request.Request(
            self._url,
            data=event.to_json().encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=self._timeout).close()
