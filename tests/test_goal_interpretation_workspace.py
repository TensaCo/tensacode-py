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


def test_deferred_selection_reuses_retained_batch_without_reenumeration(monkeypatch):
    from tensorcode.agent.goal_interpretation import select_goal
    agent, frame, batch = fixture(monkeypatch)
    deferred = resolve_goal(agent, frame, 'original retained request')
    assert isinstance(deferred.goal, Unknown)
    compared = agent.interpretations.get(deferred.group_id)
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda *a, **k: pytest.fail('deferred selection re-enumerated goals'))
    agent.goal_selector = lambda group: pytest.fail('explicit delayed decision must not invoke synchronous policy')
    selected = select_goal(agent, deferred.group_id, decision=explicit(compared))
    assert selected.group_id == deferred.group_id and selected.goal == batch.proposals[0].goal
    assert validate_dependency(agent.interpretations, selected.dependency) is True
    sources = [s for s in agent.interpretations.sources() if s.provider == 'verbnet-goal-projection']
    assert len(sources) == 1 and sources[0].text == 'original retained request'


def test_delayed_decision_requires_exact_revision_and_candidates(monkeypatch):
    from tensorcode.agent.goal_interpretation import retain_goal_proposals, select_goal
    agent, frame, _ = fixture(monkeypatch)
    group_id = retain_goal_proposals(agent, frame, 'fixture')
    group = agent.interpretations.get(group_id)
    missing_basis = InterpretationDecision(group.candidates[0].id, 'authored choice without delayed comparison basis')
    assert select_goal(agent, group_id, decision=missing_basis).goal.reason == 'goal_comparison_changed'
    old_decision = explicit(group)
    agent.interpretations.unset(group_id, reason='new comparison epoch')
    assert select_goal(agent, group_id, decision=old_decision).goal.reason == 'goal_comparison_changed'
    assert agent.interpretations.get(group_id).selected_id is None


def test_lookalike_goal_group_cannot_claim_retained_projection_authority(monkeypatch):
    from tensorcode.agent.goal_interpretation import retain_goal_proposals, select_goal
    agent, frame, _ = fixture(monkeypatch)
    original_id = retain_goal_proposals(agent, frame, 'fixture')
    original = agent.interpretations.get(original_id)
    clone = agent.interpretations.create_group(original.source_id, provenance=original.provenance)
    for candidate in original.candidates:
        agent.interpretations.propose(clone.id, candidate.payload, provenance=candidate.provenance)
    result = select_goal(agent, clone.id, decision=explicit(agent.interpretations.get(clone.id)))
    assert result.goal.reason == 'unrecognized_goal_group' and result.dependency is None


def test_retained_group_cannot_admit_extra_goal_payloads(monkeypatch):
    from tensorcode.agent.goal_interpretation import retain_goal_proposals, select_goal
    agent, frame, batch = fixture(monkeypatch)
    group_id = retain_goal_proposals(agent, frame, 'fixture')
    agent.interpretations.propose(group_id, batch.proposals[0], provenance=('authored addition outside retained batch',))
    decision = explicit(agent.interpretations.get(group_id))
    assert select_goal(agent, group_id, decision=decision).goal.reason == 'goal_group_content_changed'


def test_deferred_selection_rejects_withdrawn_parent_and_incomplete_batch(monkeypatch):
    from tensorcode.agent.goal_interpretation import retain_goal_proposals, select_goal
    agent, frame, _ = fixture(monkeypatch)
    source = agent.interpretations.add_source('authored parent')
    parent = agent.interpretations.create_group(source.id)
    candidate = agent.interpretations.propose(parent.id, {'fixture': True})
    agent.interpretations.select(parent.id, candidate.id, reason='authored parent choice')
    dependency = capture_dependency(agent.interpretations, parent.id, basis=('authored goal correspondence',))
    group_id = retain_goal_proposals(agent, frame, 'fixture', parent_dependency=dependency)
    decision = explicit(agent.interpretations.get(group_id))
    agent.interpretations.unset(parent.id, reason='new evidence withdraws parent')
    assert select_goal(agent, group_id, decision=decision).goal.reason == 'interpretation_dependency_changed'
    other, frame, _ = fixture(monkeypatch, complete=False)
    incomplete = retain_goal_proposals(other, frame, 'incomplete fixture')
    assert select_goal(other, incomplete, decision=explicit(other.interpretations.get(incomplete))).goal.reason == 'goal_search_incomplete'


def test_parent_validation_cannot_enlarge_authenticated_goal_batch(monkeypatch):
    import tensorcode.agent.goal_interpretation as boundary
    agent, frame, batch = fixture(monkeypatch)
    group_id = boundary.retain_goal_proposals(agent, frame, 'fixture')
    validate = boundary.validate_dependencies
    inserted = False
    def changing(workspace, dependencies):
        nonlocal inserted
        result = validate(workspace, dependencies)
        if not inserted:
            inserted = True
            workspace.propose(group_id, batch.proposals[0], provenance=('outside authenticated batch',))
        return result
    monkeypatch.setattr(boundary, 'validate_dependencies', changing)
    agent.goal_selector = lambda group: pytest.fail('changed authenticated batch must not reach selector')
    result = boundary.select_goal(agent, group_id)
    assert result.goal.reason == 'goal_comparison_changed' and result.dependency is None


def test_dependency_capture_cannot_bless_a_new_uncompared_goal_rival(monkeypatch):
    import tensorcode.agent.goal_interpretation as boundary
    agent, frame, batch = fixture(monkeypatch)
    agent.goal_selector = explicit
    capture = boundary.capture_dependency
    def enlarged(workspace, group_id, **kwargs):
        workspace.propose(group_id, batch.proposals[0], provenance=('new rival during goal dependency capture',))
        return capture(workspace, group_id, **kwargs)
    monkeypatch.setattr(boundary, 'capture_dependency', enlarged)
    result = boundary.resolve_goal(agent, frame, 'fixture')
    assert result.goal.reason == 'goal_comparison_changed' and result.dependency is None


def test_deferred_selection_does_not_restore_rejected_goal_candidate(monkeypatch):
    from tensorcode.agent.goal_interpretation import retain_goal_proposals, select_goal
    agent, frame, _ = fixture(monkeypatch)
    group_id = retain_goal_proposals(agent, frame, 'fixture')
    candidate = agent.interpretations.get(group_id).candidates[0]
    agent.interpretations.reject(group_id, candidate.id, reason='authored rejection before delayed choice')
    result = select_goal(agent, group_id, decision=explicit(agent.interpretations.get(group_id)))
    assert result.goal.reason == 'invalid_goal_selection'
    assert agent.interpretations.get(group_id).candidates[0].rejected
