"""Revisable goals from lexical projection or an explicitly admitted learned model.

Lexical projection is supplied semantics; learned structural correspondences retain
their teaching and validation evidence. Neither infers initial grounding or intent.
Retaining proposals never asserts beliefs or executes actions.
"""
from copy import deepcopy
from dataclasses import dataclass

from ..language import verbnet
from ..outcomes import Unknown
from ..goals import GoalSpec
from ..learning.experience import _same
from .task_dependencies import InterpretationDependency, capture_dependency, validate_dependencies


@dataclass(frozen=True)
class GoalResolution:
    goal: verbnet.Goal | GoalSpec | Unknown
    group_id: str
    dependency: InterpretationDependency | None = None
    supporting_dependencies: tuple[InterpretationDependency, ...] = ()


@dataclass(frozen=True)
class LearnedGoalProposal:
    goal: GoalSpec
    template_ids: tuple[str, ...]
    training_example_ids: tuple[str, ...]
    validation_example_ids: tuple[str, ...]
    conflicting_validation_example_ids: tuple[str, ...] = ()


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
        tuple(p.conflicting_validation_example_ids)) for p in batch.proposals)
    return LearnedGoalCandidates(proposals, deepcopy(tuple(batch.unresolved)), batch.complete), (handle.dependency,)


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
    if parent_dependency is not None and not isinstance(parent_dependency, InterpretationDependency):
        raise TypeError("parent_dependency must be an explicit InterpretationDependency")
    workspace = agent.interpretations
    learned = getattr(agent, "goal_model", None) is not None
    if learned:
        batch, supporting = _learned_candidates(agent, frame)
    else:
        batch = verbnet.goal_candidates(deepcopy(frame), agent.verbs, max_derivations=max_derivations)
        supporting = ()
    provider = "learned-goal-correspondence" if learned else "verbnet-goal-projection"
    source = workspace.add_source(source_text, modality="goal-projection", provider=provider,
        payload={"frame": deepcopy(frame), "batch": deepcopy(batch)},
        metadata={"parent_dependency": deepcopy(parent_dependency), "complete": batch.complete,
                  "projection": "supervised structural correspondence" if learned else "supplied VerbNet inventory and authored role projection",
                  "supporting_dependencies": deepcopy(supporting),
                  "max_derivations": max_derivations})
    group = workspace.create_group(source.id, provenance=("goal proposals; no implicit selection",))
    for proposal in batch.proposals:
        workspace.propose(group.id, proposal, provenance=(("learned-goal-proposal" if learned else "verbnet-goal-proposal"),))
    for unresolved in batch.unresolved:
        workspace.propose(group.id, unresolved, provenance=("unresolved-goal-projection",))

    retained = workspace.get(group.id)
    _registry(agent)[group.id] = _RetainedGoalGroup(deepcopy(source), group.provenance,
                                                  tuple(c.id for c in retained.candidates))
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
        if (source.modality != "goal-projection" or source.provider not in {"verbnet-goal-projection", "learned-goal-correspondence"}
                or not isinstance(source.payload, dict) or set(source.payload) != {"frame", "batch"}
                or not isinstance(source.payload["batch"], (verbnet.GoalCandidates, LearnedGoalCandidates))):
            return unknown("invalid_goal_group_source")
        batch = source.payload["batch"]
        learned = isinstance(batch, LearnedGoalCandidates)
        if learned != (source.provider == "learned-goal-correspondence"):
            return unknown("invalid_goal_group_source")
        expected = (*batch.proposals, *batch.unresolved)
        for index, (candidate, payload) in enumerate(zip(group.candidates, expected)):
            provenance = ((("learned-goal-proposal" if learned else "verbnet-goal-proposal"),) if index < len(batch.proposals)
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
        parent = (*(() if parent_dependency is None else (parent_dependency,)), *supporting)
    except Exception as error:
        return unknown("invalid_goal_group_source", f"{type(error).__name__}: {error}")

    def dependencies_valid():
        return validate_dependencies(workspace, parent)

    validity = dependencies_valid()
    if validity is not True:
        return unknown(validity.reason, validity.detail)
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
        if not isinstance(candidate.payload, LearnedGoalProposal if learned else verbnet.GoalProposal):
            return unknown("unresolved_goal_projection", "selected entry does not supply a goal proposal")
        proposal = candidate.payload
        if not isinstance(proposal.goal, GoalSpec if learned else verbnet.Goal):
            return unknown("invalid_goal_selection", "proposal does not contain a Goal")
        # Unmapped input roles remain explicit obligations on the returned Goal.
        # A declared refiner may consume them; ordinary lexical execution must
        # still reject any that remain. Construction mismatches cannot be repaired
        # merely by choosing a proposal or by dropping its unmatched roles.
        if not learned and proposal.derivations and not any(
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
        if selected is None or selected.id != candidate.id or not isinstance(selected.payload, LearnedGoalProposal if learned else verbnet.GoalProposal):
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
