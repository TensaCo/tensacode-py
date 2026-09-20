"""Explicit task commitments to selected interpretations, with stale-state guards.

The caller authors the mapping from meaning to goal. Capturing a dependency does
not infer that mapping, establish semantic entailment, select a candidate, assert
beliefs, or modify any goal. Empty local search frontiers are not proof that every
possible interpretation has been considered.
"""
from dataclasses import dataclass

from ..outcomes import Unknown
from .interpretation import ContinuationStatus, InterpretationWorkspace


def _strings(value, name, *, nonempty=False):
    if (type(value) is not tuple or (nonempty and not value)
            or any(type(item) is not str or not item.strip() for item in value)):
        raise ValueError(f'{name} must be an explicit tuple of nonempty strings')


@dataclass(frozen=True)
class InterpretationDependency:
    group_id: str
    source_id: str
    candidate_id: str
    revision: int
    continuation: ContinuationStatus
    candidate_ids: tuple[str, ...]
    basis: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self):
        for value in (self.group_id, self.source_id, self.candidate_id):
            if type(value) is not str or not value.strip():
                raise ValueError('interpretation dependency IDs must be nonempty strings')
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError('interpretation dependency revision must be a positive integer')
        frontier = self.continuation
        if (type(frontier) is not ContinuationStatus or type(frontier.available) is not bool
                or type(frontier.pending) is not int or frontier.pending != 0
                or type(frontier.generation) is not int
                or (frontier.generation < 1 if frontier.available else frontier.generation != 0)):
            raise ValueError('interpretation dependency requires a valid exhausted local frontier snapshot')
        _strings(self.basis, 'basis', nonempty=True)
        _strings(self.candidate_ids, 'candidate_ids', nonempty=True)
        if len(set(self.candidate_ids)) != len(self.candidate_ids) or self.candidate_id not in self.candidate_ids:
            raise ValueError('candidate IDs must be unique and include the selected candidate')
        _strings(self.evidence_ids, 'evidence_ids')
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError('evidence IDs must be unique')


def _snapshot(workspace, group_id, evidence_ids):
    basis = workspace.comparison_basis(group_id)
    group = workspace.get(group_id)
    frontier = workspace.continuation_status(group_id)
    workspace.get_source(group.source_id)
    for source_id in evidence_ids:
        workspace.get_source(source_id)
    # Source/candidate deepcopy and frontier access may call supplied code. Check
    # fresh state after reading evidence rather than trusting the initial snapshot.
    current = workspace.get(group_id)
    current_frontier = workspace.continuation_status(group_id)
    if (group.source_id != current.source_id or group.revision != current.revision
            or group.selected_id != current.selected_id
            or tuple((c.id, c.rejected) for c in group.candidates) != tuple((c.id, c.rejected) for c in current.candidates)
            or frontier != current_frontier
            or workspace.comparison_basis(group_id) != basis):
        raise ValueError('interpretation changed while its dependency was inspected')
    return current, current_frontier


def _expected_basis(dependency):
    return (dependency.source_id, dependency.revision, dependency.candidate_id,
            dependency.candidate_ids, False, dependency.continuation.available,
            dependency.continuation.generation)


def capture_dependency(workspace: InterpretationWorkspace, group_id: str, *,
                       basis: tuple[str, ...], evidence_ids: tuple[str, ...] = ()) -> InterpretationDependency:
    """Bind an authored goal interpretation to its current selected snapshot."""
    _strings(basis, 'basis', nonempty=True)
    _strings(evidence_ids, 'evidence_ids')
    group, frontier = _snapshot(workspace, group_id, evidence_ids)
    selected = group.selected
    if selected is None or selected.rejected:
        raise ValueError('capture requires a currently selected non-rejected interpretation')
    if frontier.pending:
        raise ValueError('capture requires no pending local interpretation work')
    return InterpretationDependency(group.id, group.source_id, selected.id, group.revision,
                                    frontier, tuple(c.id for c in group.candidates), basis, evidence_ids)


def validate_dependency(workspace: InterpretationWorkspace, dependency: InterpretationDependency):
    """Return True only while the exact selected interpretation remains current."""
    try:
        if type(dependency) is not InterpretationDependency:
            raise ValueError('expected InterpretationDependency')
        dependency.__post_init__()
    except Exception as error:
        return Unknown('invalid_interpretation_dependency', str(error))
    try:
        group, frontier = _snapshot(workspace, dependency.group_id, dependency.evidence_ids)
        selected = group.selected
        if (group.source_id != dependency.source_id or group.revision != dependency.revision
                or selected is None or selected.id != dependency.candidate_id or selected.rejected
                or tuple(c.id for c in group.candidates) != dependency.candidate_ids
                or frontier != dependency.continuation or frontier.pending):
            return Unknown('interpretation_dependency_changed', dependency.group_id)
    except Exception as error:
        return Unknown('interpretation_dependency_changed', f'{type(error).__name__}: {error}')
    return True


def validate_dependencies(workspace: InterpretationWorkspace, dependencies: tuple[InterpretationDependency, ...]):
    """Validate every explicitly supplied dependency; never infer missing ones."""
    if type(dependencies) is not tuple:
        return Unknown('invalid_interpretation_dependency', 'dependencies must be an explicit tuple')
    for dependency in dependencies:
        result = validate_dependency(workspace, dependency)
        if result is not True:
            return result
    # Only scalar identities are read after the final callback-bearing check.
    # Repeating full validation a fixed number of times cannot establish this:
    # its last evidence read could always revise an earlier dependency.
    for dependency in dependencies:
        try:
            if workspace.comparison_basis(dependency.group_id) != _expected_basis(dependency):
                return Unknown('interpretation_dependency_changed', dependency.group_id)
        except Exception as error:
            return Unknown('interpretation_dependency_changed', f'{type(error).__name__}: {error}')
    return True
