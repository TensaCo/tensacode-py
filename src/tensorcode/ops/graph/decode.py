"""Explicit conversion from graph representations to JSON data."""

from __future__ import annotations

import json

from ..base import Operation
from .representation import Graph, thaw_json


class JSONDecoder(Operation):
    """Decode a graph to JSON-compatible data or a JSON string."""

    replayable = True

    def __init__(self, *, as_text: bool = False):
        self.as_text = bool(as_text)

    def forward(self, value, *, context=None):
        if context:
            raise ValueError("JSONDecoder does not consume context")
        if not isinstance(value, Graph):
            raise TypeError("JSONDecoder expects a Graph")
        document = {
            "schema": "tensorcode.graph/v1",
            "identity": value.identity,
            "nodes": [
                {"id": node, "attributes": thaw_json(attributes)}
                for node, attributes in zip(value.nodes, value.node_attributes)
            ],
            "edges": [
                {
                    "source": source,
                    "relation": relation,
                    "target": target,
                    "attributes": thaw_json(attributes),
                }
                for (source, relation, target), attributes in zip(
                    value.edges, value.edge_attributes
                )
            ],
            "sources": list(value.sources),
            "source_anchors": [
                {
                    "source": anchor.source,
                    "target": anchor.target,
                    "location": thaw_json(anchor.location),
                    "attributes": thaw_json(anchor.attributes),
                }
                for anchor in value.source_anchors
            ],
            "attributes": thaw_json(value.attributes),
        }
        if self.as_text:
            return json.dumps(document, separators=(",", ":"), ensure_ascii=False)
        return document

    def configuration(self) -> dict[str, object]:
        return {"operation": "graph.json_decode", "as_text": self.as_text}
