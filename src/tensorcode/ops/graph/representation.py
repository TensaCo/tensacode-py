"""Immutable symbolic graph values with JSON-compatible extension data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any


class FrozenMap(dict[str, Any]):
    """A small recursively immutable mapping used for graph metadata."""

    __slots__ = ()

    def __init__(self, values: Mapping[str, Any] | None = None):
        values = {} if values is None else values
        if not isinstance(values, Mapping) or not all(
            isinstance(key, str) for key in values
        ):
            raise TypeError("Graph attribute maps require string keys")
        dict.__init__(self, ((key, freeze_json(value)) for key, value in values.items()))

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.items(), key=lambda item: item[0])))

    def __repr__(self) -> str:
        return f"FrozenMap({dict(self)!r})"

    def _immutable(self, *args, **kwargs):
        raise TypeError("FrozenMap values are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable


def freeze_json(value: Any) -> Any:
    """Copy a JSON value into immutable containers, rejecting opaque objects."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Graph attributes require finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        return FrozenMap(value)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    raise TypeError(f"Graph attributes require JSON values, got {type(value).__name__}")


def thaw_json(value: Any) -> Any:
    """Return ordinary JSON containers for an immutable graph value."""

    if isinstance(value, FrozenMap):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class SourceAnchor:
    """A source reference attached to a graph, node ID, or edge index."""

    source: str
    target: str | int | None = None
    location: Any = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("A source anchor requires a nonempty source reference")
        if (
            self.target is not None
            and not isinstance(self.target, str)
            and (not isinstance(self.target, int) or isinstance(self.target, bool))
        ):
            raise TypeError("An anchor target must be a node ID, edge index, or None")
        object.__setattr__(self, "location", freeze_json(self.location))
        object.__setattr__(self, "attributes", FrozenMap(self.attributes))


def _aligned_attributes(
    values: Mapping[Any, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    identities: tuple[Any, ...],
    *,
    name: str,
) -> tuple[FrozenMap, ...]:
    if isinstance(values, Mapping):
        unknown = set(values) - set(identities)
        if unknown:
            raise ValueError(f"{name} refer to unknown identities: {sorted(unknown)!r}")
        return tuple(FrozenMap(values.get(identity, {})) for identity in identities)
    items = tuple(values)
    if not items:
        return tuple(FrozenMap() for _ in identities)
    if len(items) != len(identities):
        raise ValueError(f"{name} must align one-to-one with graph entries")
    return tuple(FrozenMap(item) for item in items)


@dataclass(frozen=True)
class Graph:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str, str], ...] = ()
    sources: tuple[str, ...] = ()
    identity: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    node_attributes: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] = ()
    edge_attributes: Mapping[int, Mapping[str, Any]] | Sequence[Mapping[str, Any]] = ()
    source_anchors: tuple[SourceAnchor, ...] = ()

    def __post_init__(self) -> None:
        nodes = tuple(self.nodes)
        edges = tuple(tuple(edge) for edge in self.edges)
        sources = tuple(self.sources)
        anchors = tuple(self.source_anchors)
        if not all(isinstance(anchor, SourceAnchor) for anchor in anchors):
            raise TypeError("source_anchors must contain SourceAnchor values")
        if not all(isinstance(value, str) for value in (*nodes, *sources)):
            raise TypeError("Node identities and source references must be strings")
        if any(not node for node in nodes):
            raise ValueError("Node identities must be nonempty")
        if len(set(nodes)) != len(nodes):
            raise ValueError("Duplicate node identity")
        if len(set(sources)) != len(sources):
            raise ValueError("Duplicate source reference")
        sources = tuple(dict.fromkeys((*sources, *(anchor.source for anchor in anchors))))
        if self.identity is not None and (
            not isinstance(self.identity, str) or not self.identity
        ):
            raise ValueError("Graph identity must be a nonempty string or None")
        for edge in edges:
            if (
                len(edge) != 3
                or not all(isinstance(value, str) for value in edge)
                or edge[0] not in nodes
                or edge[2] not in nodes
            ):
                raise ValueError("Edges must connect existing nodes")
        for anchor in anchors:
            if isinstance(anchor.target, str) and anchor.target not in nodes:
                raise ValueError("Source anchor refers to an unknown node")
            if isinstance(anchor.target, int) and not 0 <= anchor.target < len(edges):
                raise ValueError("Source anchor refers to an unknown edge")

        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "attributes", FrozenMap(self.attributes))
        object.__setattr__(
            self,
            "node_attributes",
            _aligned_attributes(self.node_attributes, nodes, name="Node attributes"),
        )
        object.__setattr__(
            self,
            "edge_attributes",
            _aligned_attributes(
                self.edge_attributes,
                tuple(range(len(edges))),
                name="Edge attributes",
            ),
        )
        object.__setattr__(self, "source_anchors", anchors)

    def neighbors(self, node: str, *, relation: str | None = None) -> tuple[str, ...]:
        if node not in self.nodes:
            raise ValueError("Unknown node identity")
        return tuple(
            target
            for source, label, target in self.edges
            if source == node and (relation is None or relation == label)
        )

    def attributes_for_node(self, node: str) -> FrozenMap:
        try:
            return self.node_attributes[self.nodes.index(node)]
        except ValueError as exc:
            raise ValueError("Unknown node identity") from exc

    def attributes_for_edge(self, index: int) -> FrozenMap:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("Unknown edge index")
        try:
            return self.edge_attributes[index]
        except IndexError as exc:
            raise ValueError("Unknown edge index") from exc
