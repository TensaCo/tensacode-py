"""Revisable goal proposals from explicit lexical projection and selection policy.

The lexical inventory and projection rules are supplied semantics. Retaining or
selecting their goals does not infer grounding, assert beliefs, or execute actions.
"""
from copy import deepcopy
from dataclasses import dataclass

from ..language import verbnet
from ..outcomes import Unknown
from .task_dependencies import InterpretationDependency, capture_dependency, validate_dependencies


@dataclass(frozen=True)
class GoalResolution:
    goal: verbnet.Goal | Unknown
    group_id: str
    dependency: InterpretationDependency | None = None


def resolve_goal(agent, frame, source_text, *, parent_dependency=None, max_derivations=256) -> GoalResolution:
    """Retain every goal projection before an explicit, revision-checked choice.

    Incomplete search cannot license a choice. Unresolved entries remain visible
    alongside goal proposals; a chosen goal needs a derivation without structural
    construction obligations. Unmapped roles remain attached for explicit refiner
    consumption or execution refusal. Inventory order never supplies authority.
    """
    from .core import InterpretationDecision

    if parent_dependency is not None and not isinstance(parent_dependency, InterpretationDependency):
        raise TypeError("parent_dependency must be an explicit InterpretationDependency")
    parent = () if parent_dependency is None else (parent_dependency,)
    workspace = agent.interpretations
    batch = verbnet.goal_candidates(deepcopy(frame), agent.verbs, max_derivations=max_derivations)
    source = workspace.add_source(source_text, modality="goal-projection", provider="verbnet-goal-projection",
        payload={"frame": deepcopy(frame), "batch": deepcopy(batch)},
        metadata={"parent_dependency": deepcopy(parent_dependency), "complete": batch.complete,
                  "projection": "supplied VerbNet inventory and authored role projection",
                  "max_derivations": max_derivations})
    group = workspace.create_group(source.id, provenance=("goal proposals; no implicit selection",))
    for proposal in batch.proposals:
        workspace.propose(group.id, proposal, provenance=("verbnet-goal-proposal",))
    for unresolved in batch.unresolved:
        workspace.propose(group.id, unresolved, provenance=("unresolved-goal-projection",))

    def unknown(reason, detail=""):
        return GoalResolution(Unknown(reason, detail), group.id)

    def dependencies_valid():
        return validate_dependencies(workspace, parent)

    validity = dependencies_valid()
    if validity is not True:
        return unknown(validity.reason, validity.detail)
    if not batch.complete:
        workspace.unset(group.id, reason="goal search incomplete; no interpretation selected")
        return unknown("goal_search_incomplete")
    basis = workspace.comparison_basis(group.id)
    compared = workspace.get(group.id)
    if workspace.comparison_basis(group.id) != basis:
        return unknown("goal_comparison_changed")
    selector = getattr(agent, "goal_selector", None)
    try:
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
        if explicit_basis and (type(decision.compared_revision) is not int
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
        if not isinstance(candidate.payload, verbnet.GoalProposal):
            return unknown("unresolved_goal_projection", "selected entry does not supply a goal proposal")
        proposal = candidate.payload
        if not isinstance(proposal.goal, verbnet.Goal):
            return unknown("invalid_goal_selection", "proposal does not contain a Goal")
        # Unmapped input roles remain explicit obligations on the returned Goal.
        # A declared refiner may consume them; ordinary lexical execution must
        # still reject any that remain. Construction mismatches cannot be repaired
        # merely by choosing a proposal or by dropping its unmatched roles.
        if proposal.derivations and not any(
                all(obligation.startswith("unmapped_input_role:") for obligation in derivation.obligations)
                for derivation in proposal.derivations):
            return unknown("unresolved_goal_projection", "selected proposal retains construction obligations")
        workspace.select(group.id, candidate.id, reason=decision.reason, evidence_ids=decision.evidence_ids)
        dependency = capture_dependency(workspace, group.id,
            basis=("explicit goal interpretation selection", decision.reason), evidence_ids=decision.evidence_ids)
        selected_basis = workspace.comparison_basis(group.id)
        # Dependency capture precedes copying/extracting the selected goal. The
        # final validation catches source/goal payload callbacks that change it.
        selected = workspace.get(group.id).selected
        if selected is None or selected.id != candidate.id or not isinstance(selected.payload, verbnet.GoalProposal):
            return unknown("goal_comparison_changed")
        goal = deepcopy(selected.payload.goal)
        validity = validate_dependencies(workspace, (*parent, dependency))
        if validity is not True:
            return unknown(validity.reason, validity.detail)
        if workspace.comparison_basis(group.id) != selected_basis:
            return unknown("goal_comparison_changed")
        return GoalResolution(goal, group.id, dependency)
    except Exception as error:
        return unknown("goal_selection_error", f"{type(error).__name__}: {error}")
