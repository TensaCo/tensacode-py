"""Supplied graphs/readings test explicit uncertainty, not inferred visual truth."""
from dataclasses import replace
from types import SimpleNamespace
import pytest

from tensorcode.agent import Agent
from tensorcode.agent.core import InterpretationDecision
from tensorcode.agent.operations import Transcript
from tensorcode.agent.scene import SceneProposal
from tensorcode.agent.scene_grounding import propose_scene_groundings, UnresolvedGrounding
from tensorcode.agent.understand import Sentence
from tensorcode.language import verbnet
from tensorcode.goals import Condition, GoalSpec
from tensorcode.outcomes import Unknown
from test_agent_grounding_investigation import case, model_with_correlated_teaching, PATH
from test_agent_tasks import Devices
from agent_test_support import supplied_goal_batch, select_unique_fixture_goal


def partial_scene(agent, *, refute=False):
    model = model_with_correlated_teaching(agent)
    lg, lc, sg, _, graph = case(agent, 'partially-observed')
    a, b = graph.nodes
    facts = tuple(p for p in graph.propositions if p.roles['entity'] == a)
    if refute:
        facts += tuple(replace(p, roles={**p.roles, 'entity': b}, polarity=False) for p in facts)
    graph = replace(graph, propositions=facts)
    scene = agent.interpretations.propose(sg, SceneProposal(graph, ('incomplete scene fixture',)))
    agent.interpretations.select(sg, scene.id, reason='explicit partial scene account')
    report = propose_scene_groundings(agent, model, lg, lc, PATH, sg, scene.id)
    assert not isinstance(report, Unknown), report
    group = agent.interpretations.get(lg)
    reading = next(c for c in group.candidates if c.id == lc).payload
    sentence = Sentence('prepare the marked control', (), None, reading.acts)
    return model, lg, lc, graph, report, sentence


def retained_turn(monkeypatch, agent, lg, sentence, chosen):
    monkeypatch.setattr(agent, 'interpret', lambda text: SimpleNamespace(
        transcript=Transcript((sentence,), 'supplied retained reading'), group_ids=(lg,), unavailable=None))
    def select(group):
        return InterpretationDecision(chosen, 'explicit fixture selection after seeing all alternatives',
            compared_revision=group.revision, compared_candidate_ids=tuple(c.id for c in group.candidates))
    agent.interpretation_selector = select
    return agent.turn(sentence.text)


@pytest.mark.parametrize('unseen', [False, True])
def test_selected_unknown_referent_or_unseen_alternative_defers_turn(monkeypatch, unseen):
    devices = Devices()
    agent = Agent([devices])
    _, lg, _, graph, report, sentence = partial_scene(agent)
    assert report.complete and len(report.candidate_ids) == 1 and report.unresolved_candidate_ids
    group = agent.interpretations.get(lg)
    expected = None if unseen else graph.nodes[1]
    candidate = next(c for c in group.candidates
        if isinstance(c.payload, UnresolvedGrounding) and c.payload.reference == expected)
    monkeypatch.setattr(agent, 'handle', lambda *a, **kw: pytest.fail('unresolved reading reached dispatch'))
    turn = retained_turn(monkeypatch, agent, lg, sentence, candidate.id)
    assert turn.outcomes[0].status == 'unknown'
    assert turn.outcomes[0].candidate_id == candidate.id
    assert turn.outcomes[0].reason == candidate.payload.reason
    assert not devices.calls
    assert agent.interpretations.get(lg).selected_id == candidate.id


def test_explicit_refutation_removes_known_unknown_but_does_not_close_unseen_world():
    agent = Agent([])
    _, lg, _, graph, report, _ = partial_scene(agent, refute=True)
    source = agent.interpretations.get_source(report.source_id)
    for _, evidence in source.payload.query_evidence:
        assert next(root for root in evidence.roots if root.reference == graph.nodes[1]).status == 'refuted'
        assert evidence.unseen_referents_possible
    candidates = agent.interpretations.get(lg).candidates
    uncertainties = [c.payload for c in candidates if c.id in report.unresolved_candidate_ids]
    assert all(u.reference != graph.nodes[1] for u in uncertainties)
    assert any(u.reference is None for u in uncertainties)
    assert len(report.candidate_ids) == 1


def test_explicit_supported_choice_retains_uncertainty_in_comparison_and_can_act(monkeypatch):
    devices = Devices()
    agent = Agent([devices], goal_selector=select_unique_fixture_goal)
    model, lg, _, graph, report, sentence = partial_scene(agent)
    def goal(frame):
        return GoalSpec((Condition('enabled', {'undergoer': frame.roles['object'].ref}),))
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda frame, *a, **kw:
        supplied_goal_batch(goal(frame), frame=frame))
    monkeypatch.setattr(devices, 'refine_goal', lambda lexical: goal(lexical.frame))
    turn = retained_turn(monkeypatch, agent, lg, sentence, report.candidate_ids[0])
    outcome = turn.outcomes[0]
    assert outcome.status == 'done' and devices.calls == [graph.nodes[0]]
    task = agent.tasks.get(outcome.task_id)
    parent = next(dep for dep in task.dependencies if dep.group_id == lg)
    assert set(report.unresolved_candidate_ids) <= set(parent.candidate_ids)
    assert model.dependency in task.dependencies
    assert agent.store.propositions() == []


def test_selector_cannot_omit_unknown_alternatives_from_its_comparison(monkeypatch):
    agent = Agent([])
    _, lg, _, _, report, sentence = partial_scene(agent)
    monkeypatch.setattr(agent, 'interpret', lambda text: SimpleNamespace(
        transcript=Transcript((sentence,), 'supplied retained reading'), group_ids=(lg,), unavailable=None))
    agent.interpretation_selector = lambda group: InterpretationDecision(report.candidate_ids[0],
        'incorrectly compare only supported bindings', compared_revision=group.revision,
        compared_candidate_ids=report.candidate_ids)
    with pytest.raises(RuntimeError, match='exact comparison basis'):
        agent.turn(sentence.text)
