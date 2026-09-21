"""Explicit decoding of a JSON graph document into a graph representation."""

from __future__ import annotations

from collections.abc import Mapping
import json

from ..base import Operation
from .representation import Graph, SourceAnchor


class JSONEncoder(Operation):
    """Encode external JSON data as an immutable :class:`Graph`."""

    replayable = True

    def forward(self, value, *, context=None) -> Graph:
        if context:
            raise ValueError("JSONEncoder does not consume context")
        if isinstance(value, (str, bytes, bytearray)):
            value = json.loads(value)
        if not isinstance(value, Mapping):
            raise TypeError("JSONEncoder expects a mapping or JSON document")
        if value.get("schema") != "tensorcode.graph/v1":
            raise ValueError("Unsupported graph JSON schema")
        nodes = tuple(item["id"] for item in value.get("nodes", ()))
        edges = tuple(
            (item["source"], item["relation"], item["target"])
            for item in value.get("edges", ())
        )
        return Graph(
            nodes=nodes,
            edges=edges,
            sources=tuple(value.get("sources", ())),
            identity=value.get("identity"),
            attributes=value.get("attributes", {}),
            node_attributes=tuple(
                item.get("attributes", {}) for item in value.get("nodes", ())
            ),
            edge_attributes=tuple(
                item.get("attributes", {}) for item in value.get("edges", ())
            ),
            source_anchors=tuple(
                SourceAnchor(
                    item["source"],
                    target=item.get("target"),
                    location=item.get("location"),
                    attributes=item.get("attributes", {}),
                )
                for item in value.get("source_anchors", ())
            ),
        )

    def configuration(self) -> dict[str, str]:
        return {"operation": "graph.json_encode"}
