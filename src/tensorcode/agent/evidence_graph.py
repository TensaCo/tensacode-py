"""Source-neutral structured evidence, shared by documents and other observations.

Graph identity is an observation identity, not an image or a claim of completeness.
Visual graphs keep their existing SceneGraph specialization and anchors.
"""
from dataclasses import dataclass
import math
from numbers import Real

from ..outcomes import Score
from ..records import Proposition, Ref
from .scene import SceneGraph, _validate_value


@dataclass(frozen=True)
class EvidenceGraph:
    root: Ref
    nodes: tuple[Ref, ...] = ()
    propositions: tuple[Proposition, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self):
        self.validate()

    def validate(self):
        if type(self.root) is not Ref:
            raise TypeError('evidence graph root must be an explicit Ref')
        for name in ('nodes', 'propositions', 'limitations'):
            if type(getattr(self, name)) is not tuple:
                raise TypeError('evidence graph ' + name + ' must be a tuple')
        if any(type(node) is not Ref for node in self.nodes):
            raise TypeError('evidence graph nodes must be explicit Refs')
        if len(set(self.nodes)) != len(self.nodes) or self.root in self.nodes:
            raise ValueError('evidence graph nodes must be unique and exclude the root')
        allowed = {self.root, *self.nodes}
        for proposition in self.propositions:
            if type(proposition) is not Proposition:
                raise TypeError('evidence graph facts must be Propositions')
            try:
                _validate_value(proposition, allowed, set())
            except (ValueError, TypeError) as error:
                raise type(error)(str(error).replace('scene ', 'evidence graph ')) from error
        if any(type(item) is not str for item in self.limitations):
            raise TypeError('evidence graph limitations must be strings')


@dataclass(frozen=True)
class GraphProposal:
    """A graph interpretation naming the retained source that supplied evidence.

    Construction checks shape; workspace consumers authenticate source_id against
    their retained source record. This value alone cannot authenticate a source.
    """
    graph: EvidenceGraph
    source_id: str
    provenance: tuple[str, ...] = ()
    score: Score | None = None

    def __post_init__(self):
        self.validate()

    def validate(self):
        if type(self.graph) is not EvidenceGraph:
            raise TypeError('graph proposal requires an EvidenceGraph')
        self.graph.validate()
        if type(self.source_id) is not str or not self.source_id.strip():
            raise ValueError('graph proposal requires a nonempty retained source ID')
        if type(self.provenance) is not tuple or any(type(item) is not str for item in self.provenance):
            raise TypeError('graph provenance must be a tuple of strings')
        if self.score is not None:
            if type(self.score) is not Score:
                raise TypeError('graph score must retain its Score kind')
            if (isinstance(self.score.value, bool) or not isinstance(self.score.value, Real)
                    or not math.isfinite(self.score.value)):
                raise ValueError('graph score must be finite')
            if self.score.kind == 'probability' and not 0 <= self.score.value <= 1:
                raise ValueError('graph probability must be in [0, 1]')


def graph_root(graph):
    """Return identity for exactly the supported visual or neutral graph type."""
    if type(graph) is SceneGraph:
        return graph.image
    if type(graph) is EvidenceGraph:
        return graph.root
    raise TypeError('expected an EvidenceGraph or SceneGraph')
