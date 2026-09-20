"""Authenticated complete-task contexts for conversational association.

Association is evaluated over every task in the ledger.  Task identities and
revisions authorize that comparison but are not semantic features.  Members whose
retained originating sentence cannot be authenticated remain explicit unresolved
rivals instead of disappearing from the candidate set.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from uuid import uuid4

from ..goals import GoalSpec
from ..language import Frame
from ..learning.experience import _same
from ..outcomes import Unknown
from .scene_grounding import (
    _CandidateSnapshot,
    _capture,
    _final_comparisons,
    _validate_snapshot,
    grounding_dependencies,
)
from .task_dependencies import (
    InterpretationDependency,
    _expected_basis,
    capture_dependency,
    validate_dependencies,
)
from .task_revision_learning import _frames
from .understand import SentenceAlternative


@dataclass(frozen=True)
class TaskAssociationMember:
    """One exact task revision and its retained originating sentence, if usable."""

    task_id: str
    task_revision: int
    goal: object
    dependencies: tuple[InterpretationDependency, ...]
    goal_interpretation_id: str | None
    origin_task_revision: int | None
    origin_goal_group_id: str | None
    origin_goal_dependencies: tuple[InterpretationDependency, ...]
    origin: _CandidateSnapshot | None
    origin_dependency: InterpretationDependency | None
    origin_supporting_dependencies: tuple[InterpretationDependency, ...]
    origin_frames: tuple[Frame, ...]
    unresolved_reason: str | None = None


@dataclass(frozen=True)
class TaskAssociationContext:
    """A retained whole-sentence reading compared with one complete task set."""

    context_id: str
    incoming: _CandidateSnapshot
    incoming_dependency: InterpretationDependency
    incoming_supporting_dependencies: tuple[InterpretationDependency, ...]
    incoming_frames: tuple[Frame, ...]
    members: tuple[TaskAssociationMember, ...]
    ledger_basis: tuple[tuple[str, int], ...]
    basis: tuple[str, ...]


def _registry(agent):
    if not hasattr(agent, "_task_association_contexts"):
        agent._task_association_contexts = {}
    return agent._task_association_contexts


def _basis(value):
    if type(value) is not tuple or not value or any(type(item) is not str or not item.strip() for item in value):
        raise ValueError("association context requires an explicit nonempty basis tuple")


def _required(result, message):
    if result is not True:
        raise ValueError(f"{message}: {result.reason}: {result.detail}")


def _goal_group(agent, group_id):
    """Authenticate a retained goal group without assuming a one-frame payload."""
    retained = getattr(agent, "_goal_proposal_groups", {}).get(group_id)
    if retained is None:
        raise ValueError("unrecognized originating goal group")
    workspace = agent.interpretations
    comparison = workspace.comparison_basis(group_id)
    group = workspace.get(group_id)
    source = workspace.get_source(group.source_id)
    if (
        group.source_id != retained.source.id
        or group.provenance != retained.provenance
        or not _same(source, retained.source)
        or tuple(candidate.id for candidate in group.candidates) != retained.candidate_ids
    ):
        raise ValueError("originating goal group content changed")
    if (
        source.modality != "goal-projection"
        or source.provider
        not in {"verbnet-goal-projection", "learned-goal-correspondence", "explicit-goal-teaching"}
        or not isinstance(source.payload, dict)
        or set(source.payload) != {"frame", "batch"}
    ):
        raise ValueError("invalid originating goal source")
    batch = source.payload["batch"]
    if not hasattr(batch, "proposals") or not hasattr(batch, "unresolved"):
        raise ValueError("invalid originating goal proposal batch")
    expected = (*batch.proposals, *batch.unresolved)
    if len(expected) != len(group.candidates) or any(
        candidate.group_id != group.id or not _same(candidate.payload, payload)
        for candidate, payload in zip(group.candidates, expected)
    ):
        raise ValueError("originating goal proposal content changed")
    if source.provider == "explicit-goal-teaching":
        from .goal_interpretation import _validate_taught_parent

        _validate_taught_parent(agent, source)
    if workspace.comparison_basis(group_id) != comparison:
        raise ValueError("originating goal comparison changed during inspection")
    return group, source, comparison


def _origin(agent, task):
    """Find the earliest authenticated selected goal revision with a full parent."""
    workspace = agent.interpretations
    failures = []
    for revision in task.revisions:
        group_id = revision.goal_interpretation_id
        if group_id is None or group_id not in getattr(agent, "_goal_proposal_groups", {}):
            continue
        try:
            group, source, comparison = _goal_group(agent, group_id)
            parent = source.metadata.get("parent_dependency")
            if type(parent) is not InterpretationDependency or parent not in revision.dependencies:
                raise ValueError("originating goal lacks its selected parent dependency")
            goal_dependencies = tuple(
                dependency for dependency in revision.dependencies if dependency.group_id == group_id
            )
            if (
                group.selected is None
                or len(goal_dependencies) != 1
                or goal_dependencies[0].candidate_id != group.selected_id
                or comparison != _expected_basis(goal_dependencies[0])
            ):
                raise ValueError("task revision lacks its exact selected goal dependency")
            if (
                not hasattr(group.selected.payload, "goal")
                or not _same(group.selected.payload.goal, revision.goal)
            ):
                raise ValueError("originating goal proposal does not match the task revision goal")
            declared_support = source.metadata.get("supporting_dependencies", ())
            if (
                type(declared_support) is not tuple
                or any(type(value) is not InterpretationDependency for value in declared_support)
                or any(value not in revision.dependencies for value in declared_support)
            ):
                raise ValueError("originating goal support is incomplete")
            _required(validate_dependencies(workspace, revision.dependencies), "originating goal dependencies changed")
            snapshot = _capture(workspace, parent.group_id, parent.candidate_id, SentenceAlternative)
            if snapshot.comparison != _expected_basis(parent):
                raise ValueError("originating reading dependency changed")
            support = grounding_dependencies(agent, parent.group_id, parent.candidate_id)
            if isinstance(support, Unknown):
                raise ValueError(f"{support.reason}: {support.detail}")
            if any(value not in revision.dependencies for value in support):
                raise ValueError("originating reading grounding support is absent from the task revision")
            frames = _frames(snapshot)
            _required(validate_dependencies(workspace, revision.dependencies), "originating dependencies changed")
            _final_comparisons(workspace, (snapshot,))
            if workspace.comparison_basis(group_id) != comparison:
                raise ValueError("originating goal comparison changed during capture")
            return revision, group_id, snapshot, parent, support, frames
        except Exception as error:
            failures.append(f"revision {revision.revision}: {type(error).__name__}: {error}")
    detail = "; ".join(failures) if failures else "no retained goal revision identifies a selected parent reading"
    raise ValueError(f"origin unavailable: {detail}")


def _member(agent, task):
    unresolved = None
    if type(task.goal) is not GoalSpec:
        unresolved = "unsupported current goal; association requires a structured GoalSpec"
    validity = validate_dependencies(agent.interpretations, task.dependencies)
    if validity is not True:
        unresolved = f"current task dependencies changed: {validity.reason}: {validity.detail}"
    try:
        revision, group_id, snapshot, dependency, support, frames = _origin(agent, task)
    except Exception as error:
        if unresolved is None:
            unresolved = str(error)
        return TaskAssociationMember(
            task.id,
            task.revision,
            deepcopy(task.goal),
            deepcopy(task.dependencies),
            task.goal_interpretation_id,
            None,
            None,
            (),
            None,
            None,
            (),
            (),
            unresolved,
        )
    return TaskAssociationMember(
        task.id,
        task.revision,
        deepcopy(task.goal),
        deepcopy(task.dependencies),
        task.goal_interpretation_id,
        revision.revision,
        group_id,
        deepcopy(revision.dependencies),
        deepcopy(snapshot),
        deepcopy(dependency),
        deepcopy(support),
        deepcopy(frames),
        unresolved,
    )


def _revision(task, revision):
    return next((value for value in task.revisions if value.revision == revision), None)


def _validate_member(agent, member, *, historical):
    if type(member) is not TaskAssociationMember:
        raise ValueError("expected a task association member")
    task = agent.tasks.get(member.task_id)
    compared = _revision(task, member.task_revision) if historical else task
    if compared is None:
        raise ValueError("task association revision is absent from ledger history")
    if (
        compared.revision != member.task_revision
        or not _same(compared.goal, member.goal)
        or compared.dependencies != member.dependencies
        or compared.goal_interpretation_id != member.goal_interpretation_id
    ):
        raise ValueError("task association member changed")
    if member.unresolved_reason is not None:
        if type(member.unresolved_reason) is not str or not member.unresolved_reason.strip():
            raise ValueError("invalid unresolved task member")
        if member.origin is None:
            if (
                member.origin_task_revision is not None
                or member.origin_goal_group_id is not None
                or member.origin_goal_dependencies
                or member.origin_dependency is not None
                or member.origin_supporting_dependencies
                or member.origin_frames
            ):
                raise ValueError("unresolved origin contains partial evidence")
            return
    if (
        member.origin is None
        or type(member.origin_task_revision) is not int
        or type(member.origin_goal_group_id) is not str
        or type(member.origin_dependency) is not InterpretationDependency
        or not member.origin_frames
    ):
        raise ValueError("supported task member lacks complete origin evidence")
    historical_task = _revision(task, member.origin_task_revision)
    if (
        historical_task is None
        or historical_task.goal_interpretation_id != member.origin_goal_group_id
        or historical_task.dependencies != member.origin_goal_dependencies
    ):
        raise ValueError("originating task revision changed")
    group, source, comparison = _goal_group(agent, member.origin_goal_group_id)
    if (
        source.metadata.get("parent_dependency") != member.origin_dependency
        or group.selected is None
        or not hasattr(group.selected.payload, "goal")
        or not _same(group.selected.payload.goal, historical_task.goal)
        or comparison
        != _expected_basis(
            next(
                dependency
                for dependency in member.origin_goal_dependencies
                if dependency.group_id == member.origin_goal_group_id
            )
        )
    ):
        raise ValueError("originating goal provenance changed")
    _validate_snapshot(agent.interpretations, member.origin, SentenceAlternative)
    if not _same(_frames(member.origin), member.origin_frames):
        raise ValueError("originating complete reading changed")
    support = grounding_dependencies(
        agent, member.origin_dependency.group_id, member.origin_dependency.candidate_id
    )
    if isinstance(support, Unknown) or support != member.origin_supporting_dependencies:
        raise ValueError("originating reading support changed")
    _required(
        validate_dependencies(agent.interpretations, member.origin_goal_dependencies),
        "originating task dependencies changed",
    )
    if member.unresolved_reason is None:
        _required(validate_dependencies(agent.interpretations, member.dependencies), "task dependencies changed")


def _validate_context(agent, context, *, historical, registered):
    if type(context) is not TaskAssociationContext:
        raise ValueError("expected a retained task association context")
    _basis(context.basis)
    if type(context.context_id) is not str or not context.context_id.startswith("task-association-context:"):
        raise ValueError("invalid task association context identity")
    if registered:
        retained = _registry(agent).get(context.context_id)
        if retained is None or not _same(retained, context):
            raise ValueError("unrecognized or modified task association context")
    if (
        type(context.ledger_basis) is not tuple
        or tuple((member.task_id, member.task_revision) for member in context.members)
        != context.ledger_basis
        or len({member.task_id for member in context.members}) != len(context.members)
    ):
        raise ValueError("invalid task association ledger basis")
    if not historical and agent.tasks.comparison_basis() != context.ledger_basis:
        raise ValueError("live task membership or revision changed")
    current_basis = dict(agent.tasks.comparison_basis())
    if historical and any(current_basis.get(task_id, 0) < revision for task_id, revision in context.ledger_basis):
        raise ValueError("historical task identity or revision is unavailable")
    _validate_snapshot(agent.interpretations, context.incoming, SentenceAlternative)
    if (
        context.incoming.group.selected_id != context.incoming.candidate_id
        or context.incoming.comparison != _expected_basis(context.incoming_dependency)
        or not _same(_frames(context.incoming), context.incoming_frames)
    ):
        raise ValueError("incoming complete reading changed")
    support = grounding_dependencies(
        agent, context.incoming.group_id, context.incoming.candidate_id
    )
    if isinstance(support, Unknown) or support != context.incoming_supporting_dependencies:
        raise ValueError("incoming reading support changed")
    for member in context.members:
        _validate_member(agent, member, historical=historical)
    dependencies = (
        context.incoming_dependency,
        *context.incoming_supporting_dependencies,
        *(
            dependency
            for member in context.members
            if member.unresolved_reason is None
            for dependency in (*member.dependencies, *member.origin_goal_dependencies)
        ),
    )
    _required(
        validate_dependencies(agent.interpretations, tuple(dict.fromkeys(dependencies))),
        "context support changed",
    )
    snapshots = (
        context.incoming,
        *(member.origin for member in context.members if member.origin is not None),
    )
    _final_comparisons(agent.interpretations, snapshots)
    for dependency in tuple(dict.fromkeys(dependencies)):
        if agent.interpretations.comparison_basis(dependency.group_id) != _expected_basis(dependency):
            raise ValueError("context dependency changed during validation")
    final_basis = agent.tasks.comparison_basis()
    if not historical and final_basis != context.ledger_basis:
        raise ValueError("live task membership or revision changed during validation")
    if historical:
        final_current = dict(final_basis)
        if any(final_current.get(task_id, 0) < revision for task_id, revision in context.ledger_basis):
            raise ValueError("historical task identity or revision changed during validation")


def capture_task_association_context(agent, group_id, candidate_id, *, basis):
    """Capture a selected complete reading against every current ledger task."""
    try:
        _basis(basis)
        ledger_basis = agent.tasks.comparison_basis()
        incoming = _capture(agent.interpretations, group_id, candidate_id, SentenceAlternative)
        if incoming.group.selected_id != candidate_id:
            raise ValueError("incoming reading must be explicitly selected")
        dependency = capture_dependency(
            agent.interpretations,
            group_id,
            basis=basis,
            evidence_ids=(incoming.source.id,),
        )
        support = grounding_dependencies(agent, group_id, candidate_id)
        if isinstance(support, Unknown):
            raise ValueError(f"{support.reason}: {support.detail}")
        frames = _frames(incoming)
        members = []
        for task_id, revision in ledger_basis:
            task = agent.tasks.get(task_id)
            if task.revision != revision:
                raise ValueError("task revision changed during member capture")
            members.append(_member(agent, task))
        context = TaskAssociationContext(
            "task-association-context:" + uuid4().hex,
            incoming,
            dependency,
            support,
            frames,
            tuple(members),
            ledger_basis,
            basis,
        )
        cached, result = deepcopy(context), deepcopy(context)
        _validate_context(agent, cached, historical=False, registered=False)
        _registry(agent)[context.context_id] = cached
        return result
    except Exception as error:
        return Unknown("task_association_context_unavailable", f"{type(error).__name__}: {error}")


def validate_task_association_context(agent, context, *, historical=False):
    """Validate either the current complete comparison or its retained history.

    Historical validation accepts later ledger revisions and inserted tasks only
    for a context previously captured by this boundary.  It still authenticates
    the stored task revisions and all live interpretation support.
    """
    try:
        if type(historical) is not bool:
            raise TypeError("historical must be a boolean")
        _validate_context(agent, context, historical=historical, registered=True)
        return True
    except Exception as error:
        return Unknown("task_association_context_changed", f"{type(error).__name__}: {error}")
