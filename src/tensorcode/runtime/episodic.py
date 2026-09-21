"""Bounded source memory with caller-supplied embeddings, not a truth store."""
from __future__ import annotations

from dataclasses import dataclass
import math
from .cognitive_state import Evidence, _text


def _vector(values):
    try:
        vector = tuple(float(v) for v in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('embedding must be a numeric vector') from exc
    if not vector or not all(math.isfinite(v) for v in vector):
        raise ValueError('embedding must be nonempty and finite')
    norm = math.hypot(*vector)
    if not norm or not math.isfinite(norm):
        raise ValueError('embedding must have finite nonzero norm')
    return tuple(v / norm for v in vector)


@dataclass(frozen=True)
class RetrievalHit:
    evidence: Evidence
    score: float
    episode_id: str
    question: str = ''
    outcome: str = ''


class EpisodicMemory:
    """FIFO capacity; cosine scores indicate embedding proximity only.

    The owning tool must supply a fingerprint covering its encoder configuration
    AND weights. Changed fingerprints fail query until all vectors are rebuilt.
    """
    def __init__(self, *, capacity=256, model_fingerprint):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('capacity must be positive')
        _text(model_fingerprint, 'model_fingerprint')
        self.capacity = capacity
        self._model_fingerprint = model_fingerprint
        self._entries = {}
        self._vectors = {}

    @property
    def model_fingerprint(self):
        return self._model_fingerprint

    def __len__(self):
        return len(self._entries)

    def insert(self, evidence, vector, *, episode_id, question='', outcome='', model_fingerprint=None):
        if type(evidence) is not Evidence:
            raise ValueError('memory stores source Evidence only')
        _text(episode_id, 'episode_id')
        if not isinstance(question, str) or not isinstance(outcome, str):
            raise ValueError('metadata must be text')
        if model_fingerprint is not None and model_fingerprint != self.model_fingerprint:
            raise ValueError('stale embedding index; rebuild required')
        normalized = _vector(vector)
        if self._vectors and len(normalized) != len(next(iter(self._vectors.values()))):
            raise ValueError('embedding dimension mismatch')
        entry = RetrievalHit(evidence, 0.0, episode_id, question, outcome)
        if evidence.id in self._entries:
            if self._entries[evidence.id] != entry or self._vectors[evidence.id] != normalized:
                raise ValueError('evidence id conflicts with stored content or embedding')
            return
        entries = dict(self._entries)
        vectors = dict(self._vectors)
        entries[evidence.id] = entry
        vectors[evidence.id] = normalized
        while len(entries) > self.capacity:
            oldest = next(iter(entries))
            del entries[oldest]; del vectors[oldest]
        self._entries, self._vectors = entries, vectors

    def query(self, vector, *, model_fingerprint, k=5, exclude_episode_id=None):
        if model_fingerprint != self.model_fingerprint:
            raise ValueError('stale embedding index; rebuild required')
        if type(k) is not int or k < 0:
            raise ValueError('k must be nonnegative')
        normalized = _vector(vector)
        if self._vectors and len(normalized) != len(next(iter(self._vectors.values()))):
            raise ValueError('embedding dimension mismatch')
        hits = []
        for key, entry in self._entries.items():
            if exclude_episode_id is not None and entry.episode_id == exclude_episode_id:
                continue
            score = max(-1.0, min(1.0, math.fsum(a*b for a,b in zip(normalized,self._vectors[key]))))
            hits.append(RetrievalHit(entry.evidence, score, entry.episode_id, entry.question, entry.outcome))
        return tuple(sorted(hits, key=lambda hit: -hit.score)[:k])

    def remove(self, evidence_id):
        if evidence_id not in self._entries:
            raise KeyError(evidence_id)
        entries, vectors = dict(self._entries), dict(self._vectors)
        del entries[evidence_id]; del vectors[evidence_id]
        self._entries, self._vectors = entries, vectors

    def rebuild_index(self, vectors_by_evidence_id, *, model_fingerprint):
        _text(model_fingerprint, 'model_fingerprint')
        if set(vectors_by_evidence_id) != set(self._entries):
            raise ValueError('rebuild must provide exactly all stored evidence ids')
        vectors = {key: _vector(value) for key, value in vectors_by_evidence_id.items()}
        if len({len(value) for value in vectors.values()}) > 1:
            raise ValueError('embedding dimension mismatch')
        self._vectors = vectors
        self._model_fingerprint = model_fingerprint
