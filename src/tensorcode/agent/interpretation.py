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
        return self._record(updated, "select", candidate_id, reason)

    def unset(self, group_id: str, *, reason: str) -> InterpretationGroup:
        """Defer interpretation without rejecting the available readings."""
        group = self._groups[group_id]
        return self._record(replace(group, selected_id=None), "unset", None, reason)

    def reject(
        self, group_id: str, candidate_id: str, *, reason: str,
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
        return self._record(updated, "reject", candidate_id, reason)

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
        candidate_id: str | None, reason: str,
    ) -> InterpretationGroup:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("an interpretation decision requires a nonempty reason")
        revision = group.revision + 1
        updated = replace(
            group, revision=revision,
            history=group.history + (InterpretationRevision(
                revision, operation, candidate_id, group.selected_id, reason,
            ),),
        )
        self._groups[group.id] = updated
        return deepcopy(updated)
