"""Revisable goals from lexical projection or an explicitly admitted learned model.

Lexical projection is supplied semantics; learned structural correspondences retain
their teaching and validation evidence. Neither infers initial grounding or intent.
Retaining proposals never asserts beliefs or executes actions.
"""
from copy import deepcopy
from dataclasses import dataclass

from ..language import verbnet
from ..outcomes import Unknown
from ..goals import GoalSpec, MeasuredActionGoal
from ..learning.experience import _same
from .task_dependencies import InterpretationDependency, capture_dependency, validate_dependencies


@dataclass(frozen=True)
class GoalResolution:
    goal: verbnet.Goal | GoalSpec | MeasuredActionGoal | Unknown
    group_id: str
    dependency: InterpretationDependency | None = None
    supporting_dependencies: tuple[InterpretationDependency, ...] = ()


@dataclass(frozen=True)
class LearnedGoalProposal:
    goal: GoalSpec | MeasuredActionGoal
    template_ids: tuple[str, ...]
    training_example_ids: tuple[str, ...]
    validation_example_ids: tuple[str, ...]
    conflicting_validation_example_ids: tuple[str, ...] = ()
    conflicting_training_example_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class LearnedGoalCandidates:
    proposals: tuple[LearnedGoalProposal, ...]
    unresolved: tuple[object, ...]
    complete: bool


def _learned_candidates(agent, frame):
    from .goal_learning import get_goal_model
    handle = agent.goal_model
    model = get_goal_model(agent, handle)
    if isinstance(model, Unknown):
        return LearnedGoalCandidates((), (model,), False), ()
    batch = model.propose(deepcopy(frame))
    proposals = tuple(LearnedGoalProposal(deepcopy(p.goal), tuple(p.template_ids),
        tuple(p.training_example_ids), tuple(p.validation_example_ids),
        tuple(p.conflicting_validation_example_ids),
        tuple(p.conflicting_training_example_ids)) for p in batch.proposals)
    return LearnedGoalCandidates(proposals, deepcopy(tuple(batch.unresolved)), batch.complete), (handle.dependency,)


@dataclass(frozen=True)
class TaughtGoalProposal:
    goal: GoalSpec | MeasuredActionGoal
    reason: str


@dataclass(frozen=True)
class TaughtGoalCandidates:
    proposals: tuple[TaughtGoalProposal, ...]
    unresolved: tuple[object, ...] = ()
    complete: bool = True


def _validate_taught_parent(agent, source):
    if source.metadata.get('input_kind') == 'complete-request-sequence':
        return _validate_sentence_parent(agent, source)
    from .scene_grounding import _validate_snapshot, grounding_dependencies, _final_comparisons
    snapshot = source.metadata['parent_snapshot']
    dependency = source.metadata['parent_dependency']
    valid = validate_dependencies(agent.interpretations, (dependency, *source.metadata['supporting_dependencies']))
    if valid is not True:
        raise ValueError(valid.reason)
    selected = next(c for c in snapshot.group.candidates if c.id == snapshot.candidate_id)
    _validate_snapshot(agent.interpretations, snapshot, type(selected.payload))
    inherited = grounding_dependencies(agent, snapshot.group_id, snapshot.candidate_id)
    if isinstance(inherited, Unknown) or inherited != source.metadata['supporting_dependencies']:
        raise ValueError('teaching reading support changed')
    _final_comparisons(agent.interpretations, (snapshot,))


def _validate_sentence_parent(agent, source):
    from .scene_grounding import _validate_snapshot, grounding_dependencies, _final_comparisons
    from .task_dependencies import _expected_basis
    from .understand import SentenceAlternative
    workspace = agent.interpretations
    snapshot = source.metadata['parent_snapshot']
    parent = source.metadata['parent_dependency']
    dependencies = (parent, *source.metadata['supporting_dependencies'])
    valid = validate_dependencies(workspace, dependencies)
    if valid is not True:
        raise ValueError(valid.reason)
    _validate_snapshot(workspace, snapshot, SentenceAlternative)
    if (snapshot.group.selected_id != snapshot.candidate_id
            or snapshot.comparison != _expected_basis(parent)
            or not _same(source.payload['frame'], _request_sequence(snapshot))):
        raise ValueError('whole-request input changed')
    inherited = grounding_dependencies(agent, snapshot.group_id, snapshot.candidate_id)
    if isinstance(inherited, Unknown) or inherited != source.metadata['reading_supporting_dependencies']:
        raise ValueError('whole-request reading support changed')
    if any(dependency not in dependencies for dependency in inherited):
        raise ValueError('whole-request support is missing from goal dependencies')
    _final_comparisons(workspace, (snapshot,))
    if any(workspace.comparison_basis(d.group_id) != _expected_basis(d) for d in dependencies):
        raise ValueError('whole-request dependencies changed during validation')


def _request_sequence(snapshot):
    """Structural envelope, preserving every ordered supported request frame."""
    from ..language import Frame
    from .task_revision_learning import _frames
    return Frame('request-sequence', {'requests': _frames(snapshot)})


def retain_taught_sentence_goal(agent, goal, source_text, *, parent_dependency, reason):
    """Teach one goal for a complete request sentence without composing act goals.

    The teacher supplies the goal and constraints. The retained input contains
    every ordered request frame; this is not inferred conjunction semantics.
    """
    return _retain_taught_goal(agent, None, goal, source_text,
        parent_dependency=parent_dependency, reason=reason, whole_sentence=True)


def retain_taught_goal(agent, frame, goal, source_text, *, parent_dependency, reason):
    """Retain explicit teacher labels against one selected exact frame; select separately.

    This supplied teaching path neither enumerates lexical goals nor invokes a
    learned model. Its evidence remains distinct from learned correspondences.
    """
    return _retain_taught_goal(agent, frame, goal, source_text,
        parent_dependency=parent_dependency, reason=reason, whole_sentence=False)


def _retain_taught_goal(agent, frame, goal, source_text, *, parent_dependency, reason,
                        whole_sentence):
    from ..language import Frame
    from .understand import SentenceAlternative
    from .scene_grounding import _capture, _candidate, grounding_dependencies
    from .task_dependencies import _expected_basis
    try:
        if (not whole_sentence and type(frame) is not Frame) or type(goal) not in (GoalSpec, MeasuredActionGoal):
            raise TypeError('teaching requires an exact Frame and supported declarative goal')
        if type(parent_dependency) is not InterpretationDependency:
            raise TypeError('teaching requires an explicit selected parent dependency')
        if type(reason) is not str or not reason.strip():
            raise ValueError('teaching requires an explicit nonempty reason')
        workspace = agent.interpretations
        group = workspace.get(parent_dependency.group_id)
        if group.selected_id != parent_dependency.candidate_id or group.selected is None:
            raise ValueError('teaching parent is not explicitly selected')
        accepted = SentenceAlternative if whole_sentence else (Frame, SentenceAlternative)
        snapshot = _capture(workspace, group.id, group.selected_id, accepted)
        payload = _candidate(snapshot).payload
        if whole_sentence:
            frame = _request_sequence(snapshot)
        else:
            frames = (payload,) if type(payload) is Frame else tuple(act.frame for act in payload.acts)
            if sum(_same(frame, offered) for offered in frames) != 1:
                raise ValueError('teaching frame must exactly identify one selected parent frame')
        if snapshot.comparison != _expected_basis(parent_dependency):
            raise ValueError('teaching parent dependency changed')
        supporting = grounding_dependencies(agent, group.id, group.selected_id)
        if isinstance(supporting, Unknown):
            raise ValueError(supporting.detail)
        batch = TaughtGoalCandidates((TaughtGoalProposal(deepcopy(goal), reason),))
        source = workspace.add_source(source_text, modality='goal-projection', provider='explicit-goal-teaching',
            payload={'frame': deepcopy(frame), 'batch': deepcopy(batch)},
            metadata={'parent_dependency': deepcopy(parent_dependency), 'parent_snapshot': snapshot,
                      'supporting_dependencies': supporting, 'complete': True,
                      'reading_supporting_dependencies': supporting,
                      'input_kind': 'complete-request-sequence' if whole_sentence else 'single-frame',
                      'projection': 'explicit teacher-supplied declarative goal', 'reason': reason})
        _validate_taught_parent(agent, source)
        created = workspace.create_group(source.id, provenance=('goal proposals; no implicit selection',))
        candidate = workspace.propose(created.id, batch.proposals[0], provenance=('taught-goal-proposal',))
        cached = deepcopy(_RetainedGoalGroup(source, created.provenance, (candidate.id,)))
        _validate_taught_parent(agent, source)
        if (not _same(workspace.get_source(source.id), cached.source)
                or not _same(workspace.get(created.id).candidates[0].payload, batch.proposals[0])
                or workspace.comparison_basis(created.id) != (source.id, 0, None, (candidate.id,), None, False, 0)):
            raise ValueError('teaching evidence changed during retention')
        _registry(agent)[created.id] = cached
        return created.id
    except Exception as error:
        return Unknown('goal_teaching_unavailable', f'{type(error).__name__}: {error}')


@dataclass(frozen=True)
class _RetainedGoalGroup:
    source: object
    provenance: tuple[str, ...]
    candidate_ids: tuple[str, ...]


def _registry(agent):
    if not hasattr(agent, "_goal_proposal_groups"):
        agent._goal_proposal_groups = {}
    return agent._goal_proposal_groups


def retain_goal_proposals(agent, frame, source_text, *, parent_dependency=None, max_derivations=256) -> str:
    """Enumerate once and retain the complete supplied projection for later choice."""
    return _retain_goal_proposals(agent, frame, source_text,
        parent_dependency=parent_dependency, max_derivations=max_derivations)


def retain_sentence_goal_proposals(agent, group_id, candidate_id, *, basis):
    """Retain learned goals for a complete selected request; never infer routing."""
    from .scene_grounding import _capture, grounding_dependencies
    from .understand import SentenceAlternative
    try:
        snapshot = _capture(agent.interpretations, group_id, candidate_id, SentenceAlternative)
        dependency = capture_dependency(agent.interpretations, group_id, basis=basis)
        supporting = grounding_dependencies(agent, group_id, candidate_id)
        if isinstance(supporting, Unknown):
            return supporting
        return _retain_goal_proposals(agent, _request_sequence(snapshot), snapshot.source.text,
            parent_dependency=dependency, request_snapshot=snapshot, reading_support=supporting)
    except Exception as error:
        return Unknown('sentence_goal_projection_unavailable', f'{type(error).__name__}: {error}')


def _retain_goal_proposals(agent, frame, source_text, *, parent_dependency=None,
                           max_derivations=256, request_snapshot=None, reading_support=()):
    if parent_dependency is not None and not isinstance(parent_dependency, InterpretationDependency):
        raise TypeError("parent_dependency must be an explicit InterpretationDependency")
    workspace = agent.interpretations
    learned = getattr(agent, "goal_model", None) is not None
    if request_snapshot is not None and not learned:
        raise ValueError('whole-request projection requires an admitted learned goal model')
    if learned:
        batch, supporting = _learned_candidates(agent, frame)
    else:
        batch = verbnet.goal_candidates(deepcopy(frame), agent.verbs, max_derivations=max_derivations)
        supporting = ()
    provider = "learned-goal-correspondence" if learned else "verbnet-goal-projection"
    supporting = tuple(dict.fromkeys((*reading_support, *supporting)))
    input_metadata = ({} if request_snapshot is None else {
        'input_kind': 'complete-request-sequence', 'parent_snapshot': deepcopy(request_snapshot),
        'reading_supporting_dependencies': deepcopy(reading_support)})
    source = workspace.add_source(source_text, modality="goal-projection", provider=provider,
        payload={"frame": deepcopy(frame), "batch": deepcopy(batch)},
        metadata={"parent_dependency": deepcopy(parent_dependency), "complete": batch.complete,
                  "projection": "supervised structural correspondence" if learned else "supplied VerbNet inventory and authored role projection",
                  "supporting_dependencies": deepcopy(supporting),
                  "max_derivations": max_derivations, **input_metadata})
    if request_snapshot is not None:
        _validate_sentence_parent(agent, source)
    group = workspace.create_group(source.id, provenance=("goal proposals; no implicit selection",))
    for proposal in batch.proposals:
        workspace.propose(group.id, proposal, provenance=(("learned-goal-proposal" if learned else "verbnet-goal-proposal"),))
    for unresolved in batch.unresolved:
        workspace.propose(group.id, unresolved, provenance=("unresolved-goal-projection",))

    retained = workspace.get(group.id)
    _registry(agent)[group.id] = _RetainedGoalGroup(deepcopy(source), group.provenance,
                                                  tuple(c.id for c in retained.candidates))
    if request_snapshot is not None:
        _validate_sentence_parent(agent, source)
    return group.id


def select_goal(agent, group_id: str, *, decision=None) -> GoalResolution:
    """Select within an authenticated retained group without re-enumerating.

    A delayed caller decision must name the exact compared revision and candidate
    IDs. Configured synchronous policies may omit those fields because their input
    snapshot is captured here. Neither path can introduce new goal proposals.
    """
    from .core import InterpretationDecision

    workspace = agent.interpretations
    supplied_decision = decision is not None

    def unknown(reason, detail=""):
        return GoalResolution(Unknown(reason, detail), group_id)

    retained = _registry(agent).get(group_id)
    if retained is None:
        return unknown("unrecognized_goal_group", "group was not retained by the goal proposal boundary")
    try:
        initial_basis = workspace.comparison_basis(group_id)
        group = workspace.get(group_id)
        source = workspace.get_source(group.source_id)
        if (not _same(source, retained.source) or group.provenance != retained.provenance
                or group.source_id != retained.source.id
                or tuple(c.id for c in group.candidates) != retained.candidate_ids):
            return unknown("goal_group_content_changed")
        if (source.modality != "goal-projection" or source.provider not in {"verbnet-goal-projection", "learned-goal-correspondence", "explicit-goal-teaching"}
                or not isinstance(source.payload, dict) or set(source.payload) != {"frame", "batch"}
                or not isinstance(source.payload["batch"], (verbnet.GoalCandidates, LearnedGoalCandidates, TaughtGoalCandidates))):
            return unknown("invalid_goal_group_source")
        batch = source.payload["batch"]
        learned = isinstance(batch, LearnedGoalCandidates)
        taught = isinstance(batch, TaughtGoalCandidates)
        structured = learned or taught
        if learned != (source.provider == "learned-goal-correspondence") or taught != (source.provider == "explicit-goal-teaching"):
            return unknown("invalid_goal_group_source")
        expected = (*batch.proposals, *batch.unresolved)
        for index, (candidate, payload) in enumerate(zip(group.candidates, expected)):
            provenance = ((("taught-goal-proposal" if taught else "learned-goal-proposal" if learned else "verbnet-goal-proposal"),) if index < len(batch.proposals)
                          else ("unresolved-goal-projection",))
            if candidate.group_id != group_id or candidate.provenance != provenance or not _same(candidate.payload, payload):
                return unknown("goal_group_content_changed")
        if len(group.candidates) != len(expected) or workspace.comparison_basis(group_id) != initial_basis:
            return unknown("goal_comparison_changed")
        parent_dependency = source.metadata["parent_dependency"]
        if parent_dependency is not None and not isinstance(parent_dependency, InterpretationDependency):
            return unknown("invalid_goal_group_source", "invalid retained parent dependency")
        supporting = source.metadata.get("supporting_dependencies", ())
        if type(supporting) is not tuple or any(type(d) is not InterpretationDependency for d in supporting):
            return unknown("invalid_goal_group_source", "invalid model dependencies")
        if learned and batch.proposals and not supporting:
            return unknown("invalid_goal_group_source", "learned proposals require admitted model dependency")
        if source.metadata.get('input_kind') == 'complete-request-sequence':
            _validate_sentence_parent(agent, source)
        elif taught:
            _validate_taught_parent(agent, source)
        parent = (*(() if parent_dependency is None else (parent_dependency,)), *supporting)
    except Exception as error:
        return unknown("invalid_goal_group_source", f"{type(error).__name__}: {error}")

    def dependencies_valid():
        if taught or source.metadata.get('input_kind') == 'complete-request-sequence':
            try:
                if source.metadata.get('input_kind') == 'complete-request-sequence':
                    _validate_sentence_parent(agent, source)
                else:
                    _validate_taught_parent(agent, source)
            except Exception as error:
                return Unknown('teaching_parent_changed', str(error))
        return validate_dependencies(workspace, parent)

    validity = dependencies_valid()
    if validity is not True:
        return unknown(validity.reason, validity.detail)
    if learned and batch.unresolved:
        workspace.unset(group.id, reason="learned goal correspondence retains unresolved alternatives")
        return unknown("goal_correspondence_unresolved")
    if not batch.complete:
        workspace.unset(group.id, reason="goal search incomplete; no interpretation selected")
        return unknown("goal_search_incomplete")
    basis = initial_basis
    compared = workspace.get(group.id)
    if workspace.comparison_basis(group.id) != basis:
        return unknown("goal_comparison_changed")
    selector = getattr(agent, "goal_selector", None)
    try:
        if not supplied_decision:
            decision = (InterpretationDecision(None, "no goal selection policy supplied; goal remains unresolved")
                        if selector is None else selector(deepcopy(compared)))
    except Exception as error:
        return unknown("goal_selection_error", f"{type(error).__name__}: {error}")
    if not isinstance(decision, InterpretationDecision):
        return unknown("invalid_goal_decision", "goal_selector must return InterpretationDecision")
    if not isinstance(decision.reason, str) or not decision.reason.strip():
        return unknown("invalid_goal_decision", "goal selection requires an explicit reason")
    try:
        if type(decision.evidence_ids) is not tuple or any(type(sid) is not str for sid in decision.evidence_ids):
            return unknown("invalid_goal_decision", "evidence IDs must be an explicit tuple of source IDs")
        for evidence_id in decision.evidence_ids:
            workspace.get_source(evidence_id)
        workspace.add_source("Explicit goal interpretation policy decision", modality="assessment",
            provider="goal-selection-policy", payload=deepcopy(decision),
            metadata={"goal_group_id": group.id, "goal_source_id": source.id,
                      "compared_revision": compared.revision,
                      "compared_candidate_ids": tuple(c.id for c in compared.candidates)})
        explicit_basis = decision.compared_revision is not None or decision.compared_candidate_ids is not None
        if (supplied_decision or explicit_basis) and (type(decision.compared_revision) is not int
                               or decision.compared_revision != compared.revision
                               or decision.compared_candidate_ids != tuple(c.id for c in compared.candidates)):
            return unknown("goal_comparison_changed", "decision does not identify the compared proposal set")
        validity = dependencies_valid()
        if validity is not True:
            return unknown(validity.reason, validity.detail)
        if workspace.comparison_basis(group.id) != basis:
            return unknown("goal_comparison_changed")
        if decision.candidate_id is None:
            workspace.unset(group.id, reason=decision.reason, evidence_ids=decision.evidence_ids)
            return unknown("goal_interpretation_unresolved", decision.reason)
        candidate = next((c for c in compared.candidates if c.id == decision.candidate_id), None)
        if candidate is None or candidate.rejected:
            return unknown("invalid_goal_selection", "selected candidate is absent or rejected")
        if not isinstance(candidate.payload, TaughtGoalProposal if taught else LearnedGoalProposal if learned else verbnet.GoalProposal):
            return unknown("unresolved_goal_projection", "selected entry does not supply a goal proposal")
        proposal = candidate.payload
        if not isinstance(proposal.goal, (GoalSpec, MeasuredActionGoal) if structured else verbnet.Goal):
            return unknown("invalid_goal_selection", "proposal does not contain a Goal")
        # Unmapped input roles remain explicit obligations on the returned Goal.
        # A declared refiner may consume them; ordinary lexical execution must
        # still reject any that remain. Construction mismatches cannot be repaired
        # merely by choosing a proposal or by dropping its unmatched roles.
        if not structured and proposal.derivations and not any(
                all(obligation.startswith("unmapped_input_role:") for obligation in derivation.obligations)
                for derivation in proposal.derivations):
            return unknown("unresolved_goal_projection", "selected proposal retains construction obligations")
        selected_basis = (basis[0], compared.revision + 1, candidate.id, basis[3], False, basis[5], basis[6])
        workspace.select(group.id, candidate.id, reason=decision.reason, evidence_ids=decision.evidence_ids)
        if workspace.comparison_basis(group.id) != selected_basis:
            return unknown("goal_comparison_changed")
        dependency = capture_dependency(workspace, group.id,
            basis=("explicit goal interpretation selection", decision.reason), evidence_ids=decision.evidence_ids)
        if workspace.comparison_basis(group.id) != selected_basis:
            return unknown("goal_comparison_changed")
        # Dependency capture precedes copying/extracting the selected goal. The
        # final validation catches source/goal payload callbacks that change it.
        selected = workspace.get(group.id).selected
        if selected is None or selected.id != candidate.id or not isinstance(selected.payload, TaughtGoalProposal if taught else LearnedGoalProposal if learned else verbnet.GoalProposal):
            return unknown("goal_comparison_changed")
        goal = deepcopy(selected.payload.goal)
        validity = validate_dependencies(workspace, (*parent, dependency))
        if validity is not True:
            return unknown(validity.reason, validity.detail)
        if workspace.comparison_basis(group.id) != selected_basis:
            return unknown("goal_comparison_changed")
        return GoalResolution(goal, group.id, dependency, supporting)
    except Exception as error:
        return unknown("goal_selection_error", f"{type(error).__name__}: {error}")


def resolve_goal(agent, frame, source_text, *, parent_dependency=None, max_derivations=256) -> GoalResolution:
    """Retain fresh projections and apply only an explicitly supplied goal policy."""
    group_id = retain_goal_proposals(agent, frame, source_text, parent_dependency=parent_dependency,
                                    max_derivations=max_derivations)
    return select_goal(agent, group_id)
