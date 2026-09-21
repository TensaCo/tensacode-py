"""Optional PyTorch graph adapters.

This module is imported explicitly so importing :mod:`tensorcode.ops.graph`
does not require torch. Message passing uses graph structure and supplied node
features; edge relation strings remain correspondence metadata and are not
assigned library-authored meanings.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import torch
from torch import nn

from ..vec import Latent, Space
from ...tracing import invoke
from .representation import Graph, thaw_json


def _positive_dimension(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class GraphEncoder(nn.Module):
    """Trainable directed message passing from graph nodes to a node Latent."""

    replayable = True

    def __init__(
        self,
        input_dimensions: int,
        hidden_dimensions: int,
        output_dimensions: int,
        *,
        space: Space,
        steps: int = 2,
        feature_key: str = "features",
    ):
        super().__init__()
        self.input_dimensions = _positive_dimension(input_dimensions, "input_dimensions")
        self.hidden_dimensions = _positive_dimension(hidden_dimensions, "hidden_dimensions")
        self.output_dimensions = _positive_dimension(output_dimensions, "output_dimensions")
        if not isinstance(space, Space):
            raise TypeError("space must be a vec.Space")
        if space.dimensions != output_dimensions:
            raise ValueError("Output dimensions must match the configured vector space")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if not isinstance(feature_key, str) or not feature_key:
            raise ValueError("feature_key must be a nonempty string")
        self.space = space
        self.steps = steps
        self.feature_key = feature_key
        self.input_projection = nn.Linear(input_dimensions, hidden_dimensions)
        self.self_projections = nn.ModuleList(
            nn.Linear(hidden_dimensions, hidden_dimensions) for _ in range(steps)
        )
        self.neighbor_projections = nn.ModuleList(
            nn.Linear(hidden_dimensions, hidden_dimensions, bias=False)
            for _ in range(steps)
        )
        self.output_projection = nn.Linear(hidden_dimensions, output_dimensions)

    def _features(self, graph: Graph, supplied: torch.Tensor | None) -> torch.Tensor:
        if not graph.nodes:
            raise ValueError("GraphEncoder requires at least one node")
        if supplied is None:
            rows = []
            for node in graph.nodes:
                attributes = graph.attributes_for_node(node)
                if self.feature_key not in attributes:
                    raise ValueError(
                        f"Node {node!r} has no {self.feature_key!r} feature attribute"
                    )
                row = attributes[self.feature_key]
                if (
                    not isinstance(row, tuple)
                    or len(row) != self.input_dimensions
                    or any(
                        isinstance(item, bool) or not isinstance(item, Real)
                        for item in row
                    )
                ):
                    raise ValueError(
                        f"Node {node!r} features must contain "
                        f"{self.input_dimensions} real numbers"
                    )
                rows.append(row)
            return torch.tensor(
                rows,
                dtype=self.input_projection.weight.dtype,
                device=self.input_projection.weight.device,
            )
        if not isinstance(supplied, torch.Tensor):
            raise TypeError("node_features must be a torch.Tensor")
        expected = (len(graph.nodes), self.input_dimensions)
        if tuple(supplied.shape) != expected:
            raise ValueError(f"node_features must have shape {expected}")
        return supplied.to(
            device=self.input_projection.weight.device,
            dtype=self.input_projection.weight.dtype,
        )

    def __call__(
        self,
        graph: Graph,
        *,
        context=None,
        node_features: torch.Tensor | None = None,
    ) -> Latent:
        if node_features is not None:
            if context and "node_features" in context:
                raise ValueError("Supply node_features directly or in context, not both")
            context = {**(context or {}), "node_features": node_features}
        return invoke(self, graph, context, super().__call__)

    def forward(self, graph: Graph, *, context=None) -> Latent:
        if not isinstance(graph, Graph):
            raise TypeError("GraphEncoder expects a Graph")
        context = dict(context or {})
        unexpected = set(context) - {"node_features"}
        if unexpected:
            raise ValueError(f"Unsupported GraphEncoder context keys: {sorted(unexpected)!r}")
        node_features = context.get("node_features")
        features = self._features(graph, node_features)
        hidden = torch.relu(self.input_projection(features))
        positions = {node: index for index, node in enumerate(graph.nodes)}
        if graph.edges:
            sources = torch.tensor(
                [positions[source] for source, _, _ in graph.edges],
                dtype=torch.long,
                device=hidden.device,
            )
            targets = torch.tensor(
                [positions[target] for _, _, target in graph.edges],
                dtype=torch.long,
                device=hidden.device,
            )
        else:
            sources = targets = torch.empty(0, dtype=torch.long, device=hidden.device)
        for own, neighbor in zip(self.self_projections, self.neighbor_projections):
            aggregate = torch.zeros_like(hidden).index_add(
                0, targets, hidden.index_select(0, sources)
            )
            hidden = torch.relu(own(hidden) + neighbor(aggregate))
        output = self.output_projection(hidden)
        anchors = tuple(
            {
                "source": anchor.source,
                "target": anchor.target,
                "location": thaw_json(anchor.location),
                "attributes": thaw_json(anchor.attributes),
            }
            for anchor in graph.source_anchors
        )
        return Latent(
            output,
            self.space,
            sources=graph.sources,
            metadata={
                "node_ids": graph.nodes,
                "graph_identity": graph.identity,
                "source_anchors": anchors,
                "edge_count": len(graph.edges),
            },
        )

    def configuration(self) -> dict[str, object]:
        return {
            "operation": "graph.neural_encode",
            "input_dimensions": self.input_dimensions,
            "hidden_dimensions": self.hidden_dimensions,
            "output_dimensions": self.output_dimensions,
            "steps": self.steps,
            "feature_key": self.feature_key,
            "edge_direction": "source_to_target",
            "space": self.space.configuration(),
        }


@dataclass(frozen=True)
class GraphPrediction:
    logits: torch.Tensor
    labels: tuple[str, ...]
    node_latent: Latent
    graph_latent: Latent

    @property
    def probabilities(self) -> torch.Tensor:
        return self.logits.softmax(dim=-1)

    @property
    def value(self) -> str:
        return self.labels[int(self.logits.argmax())]


class GraphClassifier(nn.Module):
    """Mean-pool a GraphEncoder and predict one of explicit caller labels."""

    replayable = True

    def __init__(self, encoder: GraphEncoder, *, labels):
        super().__init__()
        if not isinstance(encoder, GraphEncoder):
            raise TypeError("GraphClassifier requires a GraphEncoder")
        labels = tuple(labels)
        if not labels or not all(isinstance(label, str) and label for label in labels):
            raise ValueError("labels must contain nonempty strings")
        if len(set(labels)) != len(labels):
            raise ValueError("labels must be unique")
        self.encoder = encoder
        self.labels = labels
        parameter = encoder.output_projection.weight
        self.output = nn.Linear(
            encoder.space.dimensions,
            len(labels),
            device=parameter.device,
            dtype=parameter.dtype,
        )

    def __call__(
        self,
        graph: Graph,
        *,
        context=None,
        node_features: torch.Tensor | None = None,
    ) -> GraphPrediction:
        if node_features is not None:
            if context and "node_features" in context:
                raise ValueError("Supply node_features directly or in context, not both")
            context = {**(context or {}), "node_features": node_features}
        return invoke(self, graph, context, super().__call__)

    def forward(self, graph: Graph, *, context=None) -> GraphPrediction:
        # The classifier is the traced public boundary. Calling nn.Module's
        # implementation directly retains encoder hooks while bypassing the
        # encoder's separate public trace boundary.
        nodes = nn.Module.__call__(self.encoder, graph, context=context)
        pooled_tensor = nodes.tensor.mean(dim=0)
        pooled = Latent(
            pooled_tensor,
            nodes.space,
            sources=nodes.sources,
            metadata={**nodes.metadata, "pooling": "mean"},
        )
        return GraphPrediction(self.output(pooled_tensor), self.labels, nodes, pooled)

    def configuration(self) -> dict[str, object]:
        return {
            "operation": "graph.neural_classify",
            "encoder": self.encoder.configuration(),
            "labels": list(self.labels),
            "pooling": "mean",
        }


__all__ = ["GraphClassifier", "GraphEncoder", "GraphPrediction"]
