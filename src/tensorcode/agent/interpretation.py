"""Evidence and revisable interpretations for one agent's lifetime.

A selection records which reading a caller chose to proceed with. It is neither a
claim that the reading is true nor a confidence estimate. Candidates never enter a
belief store merely by being proposed or selected here. The workspace does not
rank meanings, infer new meanings, or persist across process restarts.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class InterpretationSource:
    id: str
    text: str
    modality: str
    provider: str
    metadata: dict[str, Any] = field(default_factory=dict)
    payload: Any = None


@dataclass(frozen=True)
class InterpretationCandidate:
    id: str
    group_id: str
    payload: Any
    provenance: tuple[str, ...] = ()
    rejected: bool = False


@dataclass(frozen=True)
class InterpretationRevision:
    revision: int
    operation: str
    candidate_id: str | None
    selected_id: str | None
    reason: str
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class InterpretationGroup:
    """Mutually exclusive readings of evidence, with at most one selection."""

    id: str
    source_id: str
    provenance: tuple[str, ...] = ()
    candidates: tuple[InterpretationCandidate, ...] = ()
    selected_id: str | None = None
    revision: int = 0
    history: tuple[InterpretationRevision, ...] = ()

    @property
    def selected(self) -> InterpretationCandidate | None:
        return next((c for c in self.candidates if c.id == self.selected_id), None)


@dataclass(frozen=True)
class InterpretationExpansion:
    """A published continuation batch; search counts are not confidence."""

    group: InterpretationGroup
    candidate_ids: tuple[str, ...]
    explored: int
    pending: int


@dataclass(frozen=True)
class ContinuationStatus:
    """Owned search progress, not a claim that all meanings were enumerated."""

    available: bool
    pending: int
    generation: int


class InterpretationWorkspace:
    """Detached snapshots of evidence, alternative meanings, and decisions.

    Payloads and metadata must support ``copy.deepcopy``. Frozen records protect
    their fields; nested objects may be mutable, but returned snapshots never
    share them with workspace state. Read a fresh snapshot after any mutation.
    Source and candidate content remain unchanged; rejection and selection are
    revisable decisions retained in the group's append-only history.
    """

    def __init__(self) -> None:
        self._sources: dict[str, InterpretationSource] = {}
        self._groups: dict[str, InterpretationGroup] = {}
        self._continuations: dict[str, Any] = {}
        self._continuation_generations: dict[str, int] = {}

    def attach_continuation(self, group_id: str, continuation: Any) -> None:
        """Own an isolated reader cursor for an existing evidence group.

        Cursors live only for this workspace's lifetime and must support deepcopy.
        An attached cursor cannot be replaced, which would silently lose work.
        """
        group = self._groups[group_id]
        if group_id in self._continuations:
            raise ValueError("interpretation group already has a continuation")
        if not callable(getattr(continuation, "advance", None)):
            raise TypeError("continuation must provide advance")
        detached = deepcopy(continuation)
        self._pending(detached)
        if self._groups[group_id] is not group or group_id in self._continuations:
            raise RuntimeError("interpretation continuation changed during attachment")
        self._continuations[group_id] = detached
        self._continuation_generations[group_id] = 1

    def continuation_status(self, group_id: str) -> ContinuationStatus:
        """Inspect scalar progress without copying or exposing the owned cursor.

        Absence and local exhaustion say nothing about global interpretation
        completeness. Generation changes on attachment and committed search work.
        """
        self._groups[group_id]
        cursor = self._continuations.get(group_id)
        if cursor is None:
            return ContinuationStatus(False, 0, 0)
        return ContinuationStatus(True, self._pending(cursor),
                                  self._continuation_generations[group_id])

    def comparison_basis(self, group_id: str) -> tuple:
        """Read immutable decision identities without invoking payload callbacks.

        This deliberately neither copies source/candidate payloads nor reads a
        cursor's pending property. Official continuation updates change generation;
        callers separately validate pending work before their final basis check.
        This is a synchronous guard, not a cross-thread workspace transaction.
        """
        group = self._groups[group_id]
        selected = next((candidate for candidate in group.candidates
                         if candidate.id == group.selected_id), None)
        available = group_id in self._continuations
        return (group.source_id, group.revision, group.selected_id,
                tuple(candidate.id for candidate in group.candidates),
                selected.rejected if selected is not None else None,
                available, self._continuation_generations.get(group_id, 0))

    @staticmethod
    def _pending(cursor: Any) -> int:
        value = getattr(cursor, "pending", None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("continuation pending must be a nonnegative integer")
        return value

    def get_continuation(self, group_id: str) -> Any:
        """Return a detached cursor, or None when this group has no continuation."""
        self._groups[group_id]
        return deepcopy(self._continuations.get(group_id))

    def expand(
        self, group_id: str, *, max_expansions: int, max_candidates: int,
    ) -> InterpretationExpansion:
        """Advance and publish atomically without selecting or executing meanings.

        Work runs on a fork. Failed projection, copying, or stale-state validation
        leaves the owned cursor untouched, so generated candidates can be retried.
        New alternatives withdraw a prior selection while retaining its history.
        Progress without alternatives records a revision but preserves selection.
        """
        from .understand import SentenceAlternative

        for name, value in (("max_expansions", max_expansions), ("max_candidates", max_candidates)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        group = self._groups[group_id]
        if group_id not in self._continuations:
            raise ValueError("interpretation group has no continuation")
        owned = self._continuations[group_id]
        before_pending = self._pending(owned)
        cursor = deepcopy(owned)
        batch = cursor.advance(max_expansions=max_expansions, max_candidates=max_candidates)
        alternatives = tuple(batch.alternatives)
        if len(alternatives) > max_candidates:
            raise ValueError("continuation exceeded candidate budget")
        for name in ("explored", "pending"):
            value = getattr(batch, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"continuation {name} must be a nonnegative integer")
        if batch.explored > max_expansions:
            raise ValueError("continuation exceeded expansion budget")
        if self._pending(cursor) != batch.pending:
            raise ValueError("continuation pending disagrees with its batch")
        if any(not isinstance(alternative, SentenceAlternative) for alternative in alternatives):
            raise TypeError("continuation must produce SentenceAlternative values")
        candidates = tuple(InterpretationCandidate(
            f"reading:{uuid4().hex}", group_id, deepcopy(alternative),
            self._provenance((alternative.provenance,)),
        ) for alternative in alternatives)
        updated = replace(group, candidates=group.candidates + candidates,
                          selected_id=None if candidates else group.selected_id)
        if candidates or batch.explored:
            revision = group.revision + 1
            reason = ("new interpretation alternatives require renewed selection" if candidates
                      else "interpretation search advanced without a new alternative")
            updated = replace(updated, revision=revision, history=group.history + (
                InterpretationRevision(revision, "expand", None, updated.selected_id, reason),))
        result = InterpretationExpansion(deepcopy(updated), tuple(c.id for c in candidates),
                                         batch.explored, batch.pending)
        current = self._groups[group_id]
        if (current.revision != group.revision
                or tuple(c.id for c in current.candidates) != tuple(c.id for c in group.candidates)
                or self._continuations.get(group_id) is not owned):
            raise RuntimeError("interpretation group changed during expansion")
        self._groups[group_id] = updated
        if candidates or batch.explored or batch.pending != before_pending:
            self._continuations[group_id] = cursor
            self._continuation_generations[group_id] += 1
        return result

    def add_source(
        self, text: str, *, modality: str = "text", provider: str = "",
        metadata: dict[str, Any] | None = None, payload: Any = None,
    ) -> InterpretationSource:
        if not isinstance(text, str):
            raise TypeError("source text must be a string")
        if not isinstance(modality, str) or not modality.strip():
            raise ValueError("a source requires a nonempty modality")
        if not isinstance(provider, str):
            raise TypeError("source provider must be a string")
        source = InterpretationSource(
            f"source:{uuid4().hex}", text, modality, provider,
            deepcopy(metadata) if metadata is not None else {}, deepcopy(payload),
        )
        self._sources[source.id] = source
        return deepcopy(source)

    def get_source(self, source_id: str) -> InterpretationSource:
        return deepcopy(self._sources[source_id])

    def sources(self) -> tuple[InterpretationSource, ...]:
        return tuple(deepcopy(source) for source in self._sources.values())

    def create_group(
        self, source_id: str, *, provenance: tuple[str, ...] = (),
    ) -> InterpretationGroup:
        self._sources[source_id]  # Validate before constructing a group.
        group = InterpretationGroup(
            f"interpretation:{uuid4().hex}", source_id,
            provenance=self._provenance(provenance),
        )
        self._groups[group.id] = group
        return deepcopy(group)

    def propose(
        self, group_id: str, payload: Any, *, provenance: tuple[str, ...] = (),
    ) -> InterpretationCandidate:
        group = self._groups[group_id]
        candidate = InterpretationCandidate(
            f"reading:{uuid4().hex}", group_id, deepcopy(payload),
            self._provenance(provenance),
        )
        self._groups[group_id] = replace(
            group, candidates=group.candidates + (candidate,),
        )
        return deepcopy(candidate)

    def get(self, group_id: str) -> InterpretationGroup:
        return deepcopy(self._groups[group_id])

    def values(self) -> tuple[InterpretationGroup, ...]:
        return tuple(deepcopy(group) for group in self._groups.values())

    def select(
        self, group_id: str, candidate_id: str, *, reason: str,
        evidence_ids: tuple[str, ...] = (),
    ) -> InterpretationGroup:
        """Choose a reading; explicitly selecting a rejected reading restores it."""
        group = self._groups[group_id]
        self._candidate(group, candidate_id)
        updated = replace(
            group, selected_id=candidate_id,
            candidates=tuple(
                replace(c, rejected=False) if c.id == candidate_id else c
                for c in group.candidates
            ),
        )
        return self._record(updated, "select", candidate_id, reason, evidence_ids)

    def unset(self, group_id: str, *, reason: str, evidence_ids: tuple[str, ...] = ()) -> InterpretationGroup:
        """Defer interpretation without rejecting the available readings."""
        group = self._groups[group_id]
        return self._record(replace(group, selected_id=None), "unset", None, reason, evidence_ids)

    def reject(
        self, group_id: str, candidate_id: str, *, reason: str,
        evidence_ids: tuple[str, ...] = (),
    ) -> InterpretationGroup:
        """Retain a rejected reading and withdraw its selection if necessary."""
        group = self._groups[group_id]
        self._candidate(group, candidate_id)
        updated = replace(
            group,
            selected_id=None if group.selected_id == candidate_id else group.selected_id,
            candidates=tuple(
                replace(c, rejected=True) if c.id == candidate_id else c
                for c in group.candidates
            ),
        )
        return self._record(updated, "reject", candidate_id, reason, evidence_ids)

    @staticmethod
    def _candidate(group: InterpretationGroup, candidate_id: str) -> InterpretationCandidate:
        for candidate in group.candidates:
            if candidate.id == candidate_id:
                return candidate
        raise KeyError(candidate_id)

    @staticmethod
    def _provenance(provenance: tuple[str, ...]) -> tuple[str, ...]:
        if isinstance(provenance, str):
            raise TypeError("provenance must be a sequence of strings")
        result = tuple(provenance)
        if any(not isinstance(item, str) for item in result):
            raise TypeError("provenance must contain strings")
        return result

    def _record(
        self, group: InterpretationGroup, operation: str,
        candidate_id: str | None, reason: str, evidence_ids: tuple[str, ...] = (),
    ) -> InterpretationGroup:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("an interpretation decision requires a nonempty reason")
        evidence_ids = self._provenance(evidence_ids)
        for source_id in evidence_ids:
            self._sources[source_id]
        revision = group.revision + 1
        updated = replace(
            group, revision=revision,
            history=group.history + (InterpretationRevision(
                revision, operation, candidate_id, group.selected_id, reason, evidence_ids,
            ),),
        )
        self._groups[group.id] = updated
        return deepcopy(updated)
