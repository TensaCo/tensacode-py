"""Authored lexical proposal fixtures isolate revisable goal-selection mechanics."""
from types import SimpleNamespace

import pytest

from tensorcode.agent.core import InterpretationDecision
from tensorcode.agent.goal_interpretation import resolve_goal
from tensorcode.agent.interpretation import InterpretationWorkspace
from tensorcode.agent.task_dependencies import capture_dependency, validate_dependency
from tensorcode.language import Frame, verbnet
from tensorcode.goals import Condition, GoalSpec
from tensorcode.outcomes import Unknown


def fixture(monkeypatch, *, complete=True, obligations=(), unresolved=()):
    frame = Frame('fixture-act', {'object': 'supplied-object'}, {})
    goal = verbnet.Goal('fixture-act', 'authored-class', (Condition('exists', {'Theme': 'supplied-object'}),), frame)
    derivation = verbnet.GoalDerivation('authored-class', 0, (), (), obligations)
    batch = verbnet.GoalCandidates((verbnet.GoalProposal(goal, (derivation,)),), unresolved, complete)
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda *args, **kwargs: batch)
    return SimpleNamespace(interpretations=InterpretationWorkspace(), verbs={}, goal_selector=None), frame, batch


def explicit(group):
    return InterpretationDecision(group.candidates[0].id, 'authored choice of this supplied fixture goal',
                                  compared_revision=group.revision,
                                  compared_candidate_ids=tuple(c.id for c in group.candidates))


def test_default_defers_even_singleton_and_retains_exact_source_projection(monkeypatch):
    agent, frame, batch = fixture(monkeypatch)
    result = resolve_goal(agent, frame, 'fixture source')
    assert isinstance(result.goal, Unknown) and result.dependency is None
    assert result.goal.reason == 'goal_interpretation_unresolved'
    group = agent.interpretations.get(result.group_id)
    assert group.selected_id is None and len(group.candidates) == 1
    source = agent.interpretations.get_source(group.source_id)
    assert source.text == 'fixture source' and source.payload == {'frame': frame, 'batch': batch}
    assert source.metadata['parent_dependency'] is None
    assert group.history[-1].operation == 'unset'


def test_explicit_selection_retains_reason_evidence_and_dependency(monkeypatch):
    agent, frame, batch = fixture(monkeypatch)
    evidence = agent.interpretations.add_source('authored fixture justification')
    agent.goal_selector = lambda group: InterpretationDecision(group.candidates[0].id,
                                                              'supplied explicit correspondence', (evidence.id,))
    result = resolve_goal(agent, frame, 'fixture source')
    assert result.goal == batch.proposals[0].goal
    assert validate_dependency(agent.interpretations, result.dependency) is True
    group = agent.interpretations.get(result.group_id)
    assert group.history[-1].reason == 'supplied explicit correspondence'
    assert group.history[-1].evidence_ids == (evidence.id,)
    records = [s for s in agent.interpretations.sources() if s.provider == 'goal-selection-policy']
    assert records[-1].payload.reason == group.history[-1].reason
    assert records[-1].payload.evidence_ids == (evidence.id,)


def test_incomplete_search_retains_unresolved_entries_without_calling_selector(monkeypatch):
    unresolved = verbnet.GoalSearchUnresolved('derivation_budget')
    agent, frame, batch = fixture(monkeypatch, complete=False, unresolved=(unresolved,))
    agent.goal_selector = lambda group: pytest.fail('incomplete search must not ask policy to settle missing alternatives')
    result = resolve_goal(agent, frame, 'bounded fixture')
    assert result.goal.reason == 'goal_search_incomplete'
    group = agent.interpretations.get(result.group_id)
    assert [c.payload for c in group.candidates] == [batch.proposals[0], unresolved]
    assert group.selected_id is None


def test_projection_obligations_cannot_be_ignored_by_selection(monkeypatch):
    agent, frame, _ = fixture(monkeypatch, obligations=('construction_slots_unresolved',))
    agent.goal_selector = explicit
    result = resolve_goal(agent, frame, 'fixture')
    assert result.goal.reason == 'unresolved_goal_projection' and result.dependency is None
    assert agent.interpretations.get(result.group_id).selected_id is None
    assert any(source.provider == 'goal-selection-policy' for source in agent.interpretations.sources())


def test_selector_cannot_modify_detached_payload_to_erase_obligations(monkeypatch):
    agent, frame, _ = fixture(monkeypatch, obligations=('construction_slots_unresolved',))
    def selector(group):
        object.__setattr__(group.candidates[0].payload, 'derivations', ())
        return explicit(group)
    agent.goal_selector = selector
    assert resolve_goal(agent, frame, 'fixture').goal.reason == 'unresolved_goal_projection'


def test_changed_comparison_set_rejected_even_if_selector_acknowledges_new_set(monkeypatch):
    agent, frame, batch = fixture(monkeypatch)
    def selector(group):
        agent.interpretations.propose(group.id, batch.proposals[0], provenance=('new supplied rival',))
        return explicit(agent.interpretations.get(group.id))
    agent.goal_selector = selector
    result = resolve_goal(agent, frame, 'fixture')
    assert result.goal.reason == 'goal_comparison_changed'
    assert len(agent.interpretations.get(result.group_id).candidates) == 2


def test_parent_withdrawal_in_selector_blocks_goal_commitment(monkeypatch):
    agent, frame, _ = fixture(monkeypatch)
    source = agent.interpretations.add_source('supplied parent meaning')
    parent = agent.interpretations.create_group(source.id)
    candidate = agent.interpretations.propose(parent.id, {'authored': True})
    agent.interpretations.select(parent.id, candidate.id, reason='supplied parent selection')
    dependency = capture_dependency(agent.interpretations, parent.id, basis=('authored parent-goal correspondence',))
    def selector(group):
        agent.interpretations.unset(parent.id, reason='withdrawn parent')
        return explicit(group)
    agent.goal_selector = selector
    result = resolve_goal(agent, frame, 'fixture', parent_dependency=dependency)
    assert result.goal.reason == 'interpretation_dependency_changed' and result.dependency is None
    assert agent.interpretations.get_source(agent.interpretations.get(result.group_id).source_id).metadata['parent_dependency'] == dependency


def test_goal_extraction_callback_cannot_invalidate_captured_dependency(monkeypatch):
    import tensorcode.agent.goal_interpretation as boundary
    agent, frame, _ = fixture(monkeypatch)
    agent.goal_selector = explicit
    original = boundary.deepcopy
    def copying(value):
        result = original(value)
        if isinstance(value, verbnet.Goal):
            selected = [g for g in agent.interpretations.values() if g.selected_id]
            if selected:
                agent.interpretations.unset(selected[-1].id, reason='withdrawn during selected-goal extraction')
        return result
    monkeypatch.setattr(boundary, 'deepcopy', copying)
    result = resolve_goal(agent, frame, 'fixture')
    assert isinstance(result.goal, Unknown) and result.dependency is None
    assert result.goal.reason == 'interpretation_dependency_changed'


def test_unresolved_or_nonlexical_payload_cannot_be_selected_as_goal(monkeypatch):
    agent, frame, _ = fixture(monkeypatch, unresolved=(verbnet.GoalSearchUnresolved('unknown_semantics'),))
    agent.goal_selector = lambda group: InterpretationDecision(group.candidates[-1].id, 'explicit selection of unresolved row')
    assert resolve_goal(agent, frame, 'fixture').goal.reason == 'unresolved_goal_projection'
    batch = verbnet.GoalCandidates((verbnet.GoalProposal(GoalSpec((Condition("fixture", {}),)), ()),))
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda *a, **k: batch)
    agent.goal_selector = explicit
    assert resolve_goal(agent, frame, 'fixture').goal.reason == 'invalid_goal_selection'


def test_clean_derivation_can_support_goal_while_other_derivations_remain_visible(monkeypatch):
    agent, frame, batch = fixture(monkeypatch, obligations=('construction_slots_unresolved',))
    ambiguous = batch.proposals[0]
    clean = verbnet.GoalDerivation('authored-alternate-frame', 1, (), (), ())
    proposal = verbnet.GoalProposal(ambiguous.goal, (*ambiguous.derivations, clean))
    batch = verbnet.GoalCandidates((proposal,))
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda *a, **k: batch)
    agent.goal_selector = explicit
    result = resolve_goal(agent, frame, 'fixture')
    assert not isinstance(result.goal, Unknown)
    selected = agent.interpretations.get(result.group_id).selected
    assert selected.payload.derivations == proposal.derivations
    assert selected.payload.derivations[0].obligations


def test_unmapped_roles_remain_on_selected_goal_for_explicit_refinement(monkeypatch):
    from dataclasses import replace
    agent, frame, batch = fixture(monkeypatch, obligations=('unmapped_input_role:location',))
    frame = replace(frame, roles={**frame.roles, 'location': 'authored destination'})
    lexical = replace(batch.proposals[0].goal, frame=frame, unmapped_roles=('location',))
    proposal = verbnet.GoalProposal(lexical, batch.proposals[0].derivations)
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda *a, **k: verbnet.GoalCandidates((proposal,)))
    agent.goal_selector = explicit
    result = resolve_goal(agent, frame, 'authored refiner must explicitly consume location')
    assert isinstance(result.goal, verbnet.Goal)
    assert result.goal.unmapped_roles == ('location',)
    assert result.goal.frame.roles['location'] == 'authored destination'
    assert validate_dependency(agent.interpretations, result.dependency) is True
    selected = agent.interpretations.get(result.group_id).selected
    assert selected.payload.derivations[0].obligations == ('unmapped_input_role:location',)


def test_unmapped_roles_do_not_license_structural_construction_mismatch(monkeypatch):
    agent, frame, _ = fixture(monkeypatch, obligations=(
        'unmapped_input_role:location', 'construction_slots_unresolved'))
    agent.goal_selector = explicit
    result = resolve_goal(agent, frame, 'fixture')
    assert result.goal.reason == 'unresolved_goal_projection' and result.dependency is None
