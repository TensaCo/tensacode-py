"""Learned questions request supplied observer evidence before explicit action."""
from dataclasses import replace
import pytest

from tensorcode.agent import Agent, Plugin
from tensorcode.agent.grounding_observation import prepare_grounding_observation, observe_grounding_proposal
from tensorcode.agent.scene_grounding import (
    retain_grounding_example, fit_grounding_model, admit_grounding_model,
    propose_scene_groundings, UnresolvedGrounding,
)
from tensorcode.goals import GoalSpec, Condition
from tensorcode.language import verbnet
from tensorcode.outcomes import Unknown
from test_scene_grounding_uncertainty import language, scene, PATH
from test_agent_scene_grounding import select_bound_request
from test_agent_tasks import Devices
from agent_test_support import supplied_goal_batch, select_unique_fixture_goal


class SuppliedObserver(Plugin):
    """Authored observation semantics; no pixels decoded and no target chosen."""
    def __init__(self, answer):
        super().__init__('supplied-scene-observer')
        self.answer = answer
        self.calls = []
        self.before_observation = lambda: None
    def observe_scene_proposition(self, proposition, scene, source):
        self.before_observation()
        self.calls.append((proposition, scene, source))
        assert proposition.predicate == 'opaque-mark'
        assert proposition.roles['entity'] in scene.nodes
        assert source.modality == 'image' and source.metadata['image_ref'] == scene.image.id
        return self.answer


def initial(agent):
    teaching = []
    for name in ('train-one', 'train-two', 'heldout'):
        lg, lc = language(agent)
        sg, sc, graph = scene(agent, name, refute_b=True)
        teaching.append(retain_grounding_example(agent, lg, lc, PATH, sg, sc,
            (graph.nodes[0],), (graph.nodes[1],), basis=('explicit fixture alignment',)))
    assert not any(isinstance(r, Unknown) for r in teaching)
    model = fit_grounding_model(agent, teaching[:2], teaching[2:], max_atoms=1)
    model = admit_grounding_model(agent, model, reason='explicit fixture model admission')
    assert not isinstance(model, Unknown), model
    lg, lc = language(agent)
    sg, sc, graph = scene(agent, 'novel')
    agent.interpretations.select(sg, sc, reason='explicit partial scene interpretation')
    return model, lg, lc, sg, sc, graph


@pytest.mark.parametrize('answer', [True, False])
def test_learned_missing_question_observation_changes_grounding_before_action(monkeypatch, answer):
    observer, devices = SuppliedObserver(answer), Devices()
    agent = Agent([observer, devices], goal_selector=select_unique_fixture_goal)
    model, lg, lc, sg, sc, graph = initial(agent)
    before = propose_scene_groundings(agent, model, lg, lc, PATH, sg, sc)
    old_sentence, old_act, old_dependency = select_bound_request(agent, before, lg)
    assert any(c.payload.reference == graph.nodes[1] for c in agent.interpretations.get(lg).candidates
               if isinstance(c.payload, UnresolvedGrounding))
    proposal = prepare_grounding_observation(agent, model, lg, lc, PATH, sg, sc, graph.nodes[1])
    assert not isinstance(proposal, Unknown), proposal
    assert proposal.plan.complete and len(proposal.plan.probes) == 1
    probe = proposal.plan.probes[0]
    assert probe.proposition.roles == {'entity': graph.nodes[1]}
    assert probe.proposition.polarity is True
    positive = next(r for _, e in probe.positive_evidence for r in e.roots if r.reference == graph.nodes[1])
    negative = next(r for _, e in probe.negative_evidence for r in e.roots if r.reference == graph.nodes[1])
    assert positive.status == 'supported' and negative.status == 'refuted'
    retained_before = agent.interpretations.get_source(proposal.evidence_source_id)
    observer.before_observation = lambda: (
        agent.interpretations.get_source(proposal.evidence_source_id) == retained_before
        or pytest.fail('predictions were not retained before observation'))
    result = observe_grounding_proposal(agent, proposal, probe.id, observer)
    assert not isinstance(result, Unknown), result
    assert result.observation is answer and result.scene_candidate_id
    assert len(observer.calls) == 1 and not devices.calls
    assert agent.interpretations.get_source(proposal.evidence_source_id) == retained_before
    group = agent.interpretations.get(sg)
    assert group.selected_id == sc  # Evidence does not select its own interpretation.
    newer = next(c for c in group.candidates if c.id == result.scene_candidate_id)
    assert newer.payload.graph.propositions[:-1] == graph.propositions
    assert newer.payload.graph.propositions[-1] == replace(probe.proposition, polarity=answer)
    assert next(c for c in group.candidates if c.id == sc).payload.graph == graph

    monkeypatch.setattr(verbnet, 'goal_candidates', lambda *a, **kw: pytest.fail('old scene comparison authorized action'))
    stale = agent.request(old_sentence, old_act, [], interpretation_dependency=old_dependency)
    assert stale.status == 'unknown' and not devices.calls
    agent.interpretations.select(sg, newer.id, reason='explicit review of observed scene revision')
    after = propose_scene_groundings(agent, model, lg, lc, PATH, sg, newer.id)
    assert not isinstance(after, Unknown), after
    chosen_ref = graph.nodes[1] if answer else graph.nodes[0]
    reading = next(c for c in agent.interpretations.get(lg).candidates
        if c.id in after.candidate_ids and c.payload.acts[0].frame.roles['object'].ref == chosen_ref)
    sentence, act, dependency = select_bound_request(agent, replace(after, candidate_ids=(reading.id,)), lg)
    unknowns = [c.payload.reference for c in agent.interpretations.get(lg).candidates
                if c.id in after.unresolved_candidate_ids]
    assert graph.nodes[1] not in unknowns and None in unknowns
    def goal(frame):
        return GoalSpec((Condition('enabled', {'undergoer': frame.roles['object'].ref}),))
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda frame, *a, **kw:
        supplied_goal_batch(goal(frame), frame=frame))
    monkeypatch.setattr(devices, 'refine_goal', lambda lexical: goal(lexical.frame))
    outcome = agent.request(sentence, act, [], interpretation_dependency=dependency)
    assert outcome.status == 'done' and devices.calls == [chosen_ref]
    assert agent.store.propositions() == []


def test_observer_abstention_does_not_refute_or_rewrite_partial_scene():
    observer = SuppliedObserver(Unknown('not_observable', 'fixture observer lacks this evidence'))
    agent = Agent([observer])
    model, lg, lc, sg, sc, graph = initial(agent)
    proposal = prepare_grounding_observation(agent, model, lg, lc, PATH, sg, sc, graph.nodes[1])
    assert not isinstance(proposal, Unknown), proposal
    before = agent.interpretations.get(sg)
    result = observe_grounding_proposal(agent, proposal, proposal.plan.probes[0].id, observer)
    assert not isinstance(result, Unknown), result
    assert isinstance(result.observation, Unknown) and result.scene_candidate_id is None
    assert agent.interpretations.get(sg) == before
    report = propose_scene_groundings(agent, model, lg, lc, PATH, sg, sc)
    unknowns = [c.payload.reference for c in agent.interpretations.get(lg).candidates
               if c.id in report.unresolved_candidate_ids]
    assert graph.nodes[1] in unknowns and None in unknowns
