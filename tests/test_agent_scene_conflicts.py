"""Supplied contradictory scene evidence must not become executable grounding."""
from dataclasses import replace
import pytest

from tensorcode.agent import Agent
from tensorcode.agent.scene import SceneProposal
from tensorcode.agent.scene_grounding import (
    propose_scene_groundings, retain_grounding_example, fit_grounding_model,
    admit_grounding_model,
)
from tensorcode.agent.grounding_investigation import (
    prepare_grounding_investigation, record_grounding_feedback,
)
from tensorcode.language import verbnet
from tensorcode.goals import Condition, GoalSpec
from tensorcode.outcomes import Unknown
from test_agent_grounding_investigation import case, model_with_correlated_teaching, PATH
from test_agent_scene_grounding import select_bound_request
from test_agent_tasks import Devices
from agent_test_support import supplied_goal_batch, select_unique_fixture_goal


def contradict(agent, sg, graph):
    opposing = tuple(replace(p, polarity=not p.polarity) for p in graph.propositions
                     if p.roles['value'] is True)
    revised = replace(graph, propositions=(*graph.propositions, *opposing))
    child = agent.interpretations.propose(sg, SceneProposal(revised, ('explicit conflicting visual fixture',)))
    agent.interpretations.select(sg, child.id, reason='explicitly compare this contradictory scene')
    return child.id, revised


def test_conflicting_scene_blocks_grounding_and_old_request_until_explicit_revision(monkeypatch):
    devices = Devices()
    agent = Agent([devices], goal_selector=select_unique_fixture_goal)
    model = model_with_correlated_teaching(agent)
    lg, lc, sg, sc, graph = case(agent, 'new-scene')
    original = propose_scene_groundings(agent, model, lg, lc, PATH, sg, sc)
    sentence, act, dependency = select_bound_request(agent, original, lg)
    conflict_id, conflicting = contradict(agent, sg, graph)
    blocked = propose_scene_groundings(agent, model, lg, lc, PATH, sg, conflict_id)
    assert not isinstance(blocked, Unknown), blocked
    assert not blocked.complete and not blocked.candidate_ids
    assert any('contradictory_match_evidence' in reason for reason in blocked.unresolved)
    evidence = agent.interpretations.get_source(blocked.source_id)
    assert evidence.payload.matches and all(match.conflicts for match in evidence.payload.matches)
    assert any((0, (4,)) in match.conflicts for match in evidence.payload.matches)
    assert evidence.metadata['scene'].group.candidates[-1].payload.graph == conflicting
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda *a, **kw: pytest.fail('withdrawn scene reached goals'))
    result = agent.request(sentence, act, [], interpretation_dependency=dependency)
    assert result.status == 'unknown' and not devices.calls

    # Selecting another retained account is an explicit fixture decision, not
    # automatic conflict resolution or a claim that the original account is true.
    agent.interpretations.select(sg, sc, reason='teacher explicitly returns to the coherent scene account')
    revised = propose_scene_groundings(agent, model, lg, lc, PATH, sg, sc)
    sentence, act, dependency = select_bound_request(agent, revised, lg)
    def goal(frame):
        return GoalSpec((Condition('enabled', {'undergoer': frame.roles['object'].ref}),))
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda frame, *a, **kw:
        supplied_goal_batch(goal(frame), frame=frame))
    monkeypatch.setattr(devices, 'refine_goal', lambda lexical: goal(lexical.frame))
    outcome = agent.request(sentence, act, [], interpretation_dependency=dependency)
    assert outcome.status == 'done' and devices.calls == [graph.nodes[0]]
    assert agent.store.propositions() == []


def test_conflicting_prediction_is_retained_but_cannot_rank_or_accept_feedback():
    agent = Agent([])
    model = model_with_correlated_teaching(agent)
    lg, lc, sg, _, graph = case(agent, 'crossed-conflict', color=0, shape=1)
    sc, _ = contradict(agent, sg, graph)
    proposal = prepare_grounding_investigation(agent, model, lg, lc, PATH, ((sg, sc),))
    assert not isinstance(proposal, Unknown), proposal
    assert not proposal.investigation.complete and not proposal.investigation.best_scene_ids
    predictions = proposal.investigation.predictions[0].predictions
    assert any(match.conflicts for p in predictions for match in p.matches)
    feedback = record_grounding_feedback(agent, proposal, graph.image,
        (graph.nodes[0],), (graph.nodes[1],), basis=('labels cannot erase contradictory visual evidence',))
    assert isinstance(feedback, Unknown)
    assert agent.interpretations.get(model.group_id).selected_id == model.candidate_id


def test_conflicting_training_is_diagnostic_and_model_cannot_be_admitted():
    agent = Agent([])
    records = []
    for name in ('first', 'second', 'heldout'):
        lg, lc, sg, sc, graph = case(agent, name)
        if name == 'first':
            sc, graph = contradict(agent, sg, graph)
        record = retain_grounding_example(agent, lg, lc, PATH, sg, sc,
            (graph.nodes[0],), (graph.nodes[1],), basis=('explicit target teaching on retained graph',))
        assert not isinstance(record, Unknown), record
        records.append(record)
    model = fit_grounding_model(agent, records[:2], records[2:], max_patterns=2048, max_matches=20000)
    assert not isinstance(model, Unknown), model
    source = agent.interpretations.get_source(model.evidence_source_id)
    assert source.payload['_unresolved'] and not source.payload['_complete']
    admitted = admit_grounding_model(agent, model, reason='attempt to admit conflicted visual support')
    assert isinstance(admitted, Unknown)
