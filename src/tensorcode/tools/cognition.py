"""Bounded, immutable session data. Records do not infer facts or execute policies."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping


def _text(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f'{name} must be a nonempty string')


@dataclass(frozen=True)
class Evidence:
    id: str
    text: str
    source_id: str

    def __post_init__(self):
        for name in ('id', 'text', 'source_id'):
            _text(getattr(self, name), name)


@dataclass(frozen=True)
class Hypothesis:
    id: str
    text: str
    origin: str = 'generated'
    # Keyword-only so provenance is always stated, including for supplied text.
    model_provenance: str = field(kw_only=True)

    def __post_init__(self):
        _text(self.id, 'id'); _text(self.text, 'text')
        if self.origin not in ('generated', 'supplied'):
            raise ValueError('origin must be generated or supplied')
        _text(self.model_provenance, 'model_provenance')


@dataclass(frozen=True)
class Assessment:
    evidence_id: str
    hypothesis_id: str
    scores: Mapping[str, float]
    model_provenance: str
    revision: int = 0

    def __post_init__(self):
        for name in ('evidence_id', 'hypothesis_id', 'model_provenance'):
            _text(getattr(self, name), name)
        if not isinstance(self.scores, Mapping) or not self.scores:
            raise ValueError('scores must be a nonempty mapping')
        for label, score in self.scores.items():
            _text(label, 'score label')
            if type(score) not in (int, float) or not math.isfinite(score):
                raise ValueError('scores must be finite numbers')
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError('invalid assessment revision')
        object.__setattr__(self, 'scores', MappingProxyType(dict(self.scores)))


@dataclass(frozen=True)
class Goal:
    id: str
    text: str
    source_id: str

    def __post_init__(self):
        for name in ('id', 'text', 'source_id'):
            _text(getattr(self, name), name)


@dataclass(frozen=True)
class Plan:
    id: str
    steps: tuple[str, ...]
    predicted_outcomes: tuple[str, ...]

    def __post_init__(self):
        _text(self.id, 'id')
        for name in ('steps', 'predicted_outcomes'):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)):
                raise ValueError(f'{name} must be a sequence')
            for value in values:
                _text(value, name)
            object.__setattr__(self, name, tuple(values))


@dataclass(frozen=True)
class Observation:
    """Externally supplied actual feedback, never generated from hypotheses."""
    id: str
    text: str
    source_id: str

    def __post_init__(self):
        for name in ('id', 'text', 'source_id'):
            _text(getattr(self, name), name)


@dataclass(frozen=True)
class RetrievalHit:
    evidence: Evidence
    score: float
    episode_id: str
    question: str = ''
    outcome: str = ''



__all__ = ["Evidence", "Hypothesis", "Assessment", "Goal", "Plan", "Observation", "RetrievalHit"]
