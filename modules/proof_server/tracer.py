"""A thin recorder scenarios call to emit canonical events.

The tracer does NO FLOPs math (that is build_graph's job -- DRY). It just
collects events with auto-incrementing ids and the model shapes they reference,
and hands back a canonical ``trace`` dict.
"""
from __future__ import annotations

from dataclasses import asdict

from modules.proof_server.graph import Event


class Tracer:
    def __init__(self) -> None:
        self._events: list[Event] = []
        self._shapes: dict[str, dict] = {}

    def add_shape(self, key: str, config: dict) -> None:
        """Register the config dict for a model key referenced by events."""
        self._shapes[key] = config

    def event(self, kind: str, *, inputs=(), model="", tokens=0, attended=0,
              mode="fwd", logits=0, status="", label="", payload=None) -> int:
        """Record one event; returns its new id."""
        eid = len(self._events)
        self._events.append(Event(
            id=eid, kind=kind, inputs=list(inputs), model=model, tokens=tokens,
            attended=attended, mode=mode, logits=logits, status=status,
            label=label, payload=payload,
        ))
        return eid

    def trace(self) -> dict:
        """Return the canonical trace dict (events as plain dicts)."""
        return {"shapes": dict(self._shapes),
                "events": [asdict(e) for e in self._events]}
