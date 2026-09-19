"""Source-bound visual interpretations, without a fixed visual relation ontology.

A scene may describe a whole image, regions, entities, relationships, events, or
nested hypotheses using the same propositions as the rest of the agent. Graphs
are proposals, not assertions in the world store. A graph need not claim that it
is complete; limitations are retained alongside its structured content.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from numbers import Real
from typing import Any

from ..outcomes import Score
from ..records import MODALITIES, Interval, Proposition, Ref, Var, matches


@dataclass(frozen=True)
class VisualAnchor:
    """An entity's support in its graph's image; ``None`` means the whole image.

    A region is a normalized ``(left, top, right, bottom)`` rectangle. Anchors
    locate evidence; they do not imply object detection or exhaustive segmentation.
    """

    entity: Ref
    region: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.entity, Ref):
            raise TypeError("anchor entity must be a Ref")
        if self.region is None:
            return
        if not isinstance(self.region, tuple) or len(self.region) != 4:
            raise ValueError("anchor region must be a normalized xyxy tuple")
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
               or not 0 <= v <= 1 for v in self.region):
            raise ValueError("anchor coordinates must be finite numbers in [0, 1]")
        left, top, right, bottom = self.region
        if left >= right or top >= bottom:
            raise ValueError("anchor region must have positive width and height")


@dataclass(frozen=True)
class SceneGraph:
    """One possible interpretation bound to one image's identity.

    Nodes have no prescribed kinds. Predicates and role names are domain data.
    Every reference, including nested proposition scopes, must resolve locally to
    ``image`` or a declared node. This establishes referential integrity, not the
    truth of an interpretation or alignment to entities in another observation.
    """

    image: Ref
    nodes: tuple[Ref, ...] = ()
    propositions: tuple[Proposition, ...] = ()
    anchors: tuple[VisualAnchor, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Recheck boundaries, including mutable payloads inside propositions."""
        if not isinstance(self.image, Ref):
            raise TypeError("scene image must be a Ref")
        for name in ("nodes", "propositions", "anchors", "limitations"):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError(f"scene {name} must be a tuple")
        if any(not isinstance(node, Ref) for node in self.nodes):
            raise TypeError("scene nodes must be Refs")
        if len(set(self.nodes)) != len(self.nodes) or self.image in self.nodes:
            raise ValueError("scene nodes must be unique and exclude the image")
        allowed = {self.image, *self.nodes}
        for proposition in self.propositions:
            if not isinstance(proposition, Proposition):
                raise TypeError("scene propositions must be Propositions")
            _validate_value(proposition, allowed, set())
        for anchor in self.anchors:
            if not isinstance(anchor, VisualAnchor):
                raise TypeError("scene anchors must be VisualAnchors")
            anchor.validate()
            if anchor.entity not in allowed:
                raise ValueError(f"anchor references undeclared entity {anchor.entity}")
        if any(not isinstance(item, str) for item in self.limitations):
            raise TypeError("scene limitations must be strings")

    def match(self, pattern: Proposition) -> tuple[dict[str, Any], ...]:
        """Query this proposal with the ordinary records matching semantics.

        An omitted pattern scope is a wildcard, as in ``records.matches``. This
        query neither asserts the matched content nor treats it as verified.
        """
        self.validate()
        if not isinstance(pattern, Proposition):
            raise TypeError("scene query must be a Proposition")
        return tuple(binding for fact in self.propositions
                     if (binding := matches(pattern, fact)) is not None)


@dataclass(frozen=True)
class SceneProposal:
    """A scene interpretation with its producer's provenance and optional score.

    The score retains its declared kind; model confidence is not automatically
    calibrated probability or confidence in every proposition in the graph.
    """

    graph: SceneGraph
    provenance: tuple[str, ...] = ()
    score: Score | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.graph, SceneGraph):
            raise TypeError("scene proposal graph must be a SceneGraph")
        self.graph.validate()
        if not isinstance(self.provenance, tuple) or any(
            not isinstance(item, str) for item in self.provenance
        ):
            raise TypeError("scene provenance must be a tuple of strings")
        if self.score is not None:
            if not isinstance(self.score, Score):
                raise TypeError("scene score must retain its Score kind")
            if (isinstance(self.score.value, bool) or not isinstance(self.score.value, Real)
                    or not math.isfinite(self.score.value)):
                raise ValueError("scene score must be finite")
            if self.score.kind == "probability" and not 0 <= self.score.value <= 1:
                raise ValueError("scene probability must be in [0, 1]")


def _validate_value(value: Any, allowed: set[Ref], active: set[int]) -> None:
    if isinstance(value, Ref):
        if value not in allowed:
            raise ValueError(f"scene references undeclared entity {value}")
        return
    if isinstance(value, Var) or value is None:
        raise ValueError("scene propositions must have bound fillers")
    if isinstance(value, (str, bool, int, float, date, datetime, Enum)):
        return
    ident = id(value)
    if ident in active:
        raise ValueError("scene proposition payloads must not contain cycles")
    active.add(ident)
    try:
        if isinstance(value, Proposition):
            if not isinstance(value.predicate, str) or not value.predicate.strip():
                raise ValueError("scene propositions need a nonempty predicate")
            if not isinstance(value.roles, Mapping) or any(
                not isinstance(key, str) or not key.strip() for key in value.roles
            ):
                raise ValueError("scene proposition roles need nonempty names")
            if not isinstance(value.polarity, bool) or value.modality not in MODALITIES:
                raise ValueError("invalid scene proposition polarity or modality")
            if not isinstance(value.valid, Interval):
                raise TypeError("scene proposition validity must be an Interval")
            if value.scope is not None:
                if not isinstance(value.scope, Ref):
                    raise TypeError("scene proposition scope must be a Ref")
                _validate_value(value.scope, allowed, active)
            for filler in value.roles.values():
                _validate_value(filler, allowed, active)
        elif isinstance(value, Mapping):
            for key, filler in value.items():
                _validate_value(key, allowed, active)
                _validate_value(filler, allowed, active)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for filler in value:
                _validate_value(filler, allowed, active)
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                filler = getattr(value, field.name)
                # Optional fields in typed literal values are not missing roles.
                if filler is not None:
                    _validate_value(filler, allowed, active)
        else:
            raise TypeError(f"unsupported scene filler {type(value).__name__}")
    finally:
        active.remove(ident)
