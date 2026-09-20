"""Learned relational questions consume authored scenes and observer evidence."""
from dataclasses import replace

import pytest

from tensorcode.agent import Agent, Plugin
from tensorcode.agent.scene import SceneGraph, SceneProposal
from tensorcode.agent.grounding_observation import prepare_grounding_observation, observe_grounding_proposal
from tensorcode.agent.scene_grounding import (
    retain_grounding_example, fit_grounding_model, admit_grounding_model,
    propose_scene_groundings, get_grounding_model,
)
from tensorcode.goals import GoalSpec, Condition
from tensorcode.language import verbnet
from tensorcode.outcomes import Unknown
from tensorcode.records import Ref, Proposition
from test_scene_grounding_uncertainty import language, PATH
from test_agent_scene_grounding import select_bound_request
from test_agent_tasks import Devices
from agent_test_support import supplied_goal_batch, select_unique_fixture_goal


def scene(agent, name, *, partial=False):
    image, a, b, g, h = (Ref(name + ':' + label) for label in ('image', 'a', 'b', 'g', 'h'))
    facts = [Proposition('member', {'item': a, 'group': g}),
             Proposition('member', {'item': b, 'group': h}),
             Proposition('radial', {'group': h}, polarity=False)]
    if not partial:
        facts.append(Proposition('radial', {'group': g}))
    graph = SceneGraph(image, (a, b, g, h), tuple(facts))
    source = agent.interpretations.add_source('supplied relational scene', modality='image',
        payload=b'authored scene fixture, not decoded pixels', metadata={'image_ref': image.id})
    group = agent.interpretations.create_group(source.id)
    candidate = agent.interpretations.propose(group.id, SceneProposal(graph))
    return group.id, candidate.id, graph


def setup(agent):
    examples = []
    for name in ('train-one', 'train-two', 'validation'):
        lg, lc = language(agent)
        sg, sc, graph = scene(agent, name)
        record = retain_grounding_example(agent, lg, lc, PATH, sg, sc,
            (graph.nodes[0],), graph.nodes[1:], basis=('authored scene alignment',))
        assert not isinstance(record, Unknown), record
        examples.append(record)
    handle = fit_grounding_model(agent, examples[:2], examples[2:], max_atoms=2)
    assert not isinstance(handle, Unknown), handle
    handle = admit_grounding_model(agent, handle, reason='explicit heldout model admission')
    assert not isinstance(handle, Unknown), handle
    learned = get_grounding_model(agent, handle)
    assert learned.queries and all(len(q.query.atoms) == 2 for q in learned.queries)
    lg, lc = language(agent)
    sg, sc, graph = scene(agent, 'novel', partial=True)
    agent.interpretations.select(sg, sc, reason='explicit partial scene choice')
    return handle, lg, lc, sg, sc, graph


class Observer(Plugin):
    def __init__(self, answer):
        super().__init__('supplied-organization-observer')
        self.answer, self.calls = answer, []

    def observe_scene_proposition(self, proposition, graph, source):
        self.calls.append(proposition)
        assert proposition == Proposition('radial', {'group': graph.nodes[2]})
        assert source.payload == b'authored scene fixture, not decoded pixels'
        return self.answer


@pytest.mark.parametrize('answer', [True, False])
def test_learned_relation_derives_question_about_existing_group_before_action(monkeypatch, answer):
    observer, devices = Observer(answer), Devices()
    agent = Agent([observer, devices], goal_selector=select_unique_fixture_goal)
    model, lg, lc, sg, sc, graph = setup(agent)
    before = propose_scene_groundings(agent, model, lg, lc, PATH, sg, sc)
    assert not isinstance(before, Unknown) and not before.candidate_ids
    proposal = prepare_grounding_observation(agent, model, lg, lc, PATH, sg, sc, graph.nodes[0])
    assert not isinstance(proposal, Unknown), proposal
    assert proposal.plan.complete and len(proposal.plan.probes) == 1
    probe = proposal.plan.probes[0]
    assert probe.proposition == Proposition('radial', {'group': graph.nodes[2]})
    assert probe.witnesses
    assert all(graph.nodes[2] in dict(w.bindings).values() for w in probe.witnesses)
    assert all(any(fact == 0 for _, fact in w.supporting) for w in probe.witnesses)
    for _, evidence in probe.positive_evidence:
        assert next(r.status for r in evidence.roots if r.reference == graph.nodes[0]) == 'supported'
    for _, evidence in probe.negative_evidence:
        assert next(r.status for r in evidence.roots if r.reference == graph.nodes[0]) == 'unknown'
    retained = agent.interpretations.get_source(proposal.evidence_source_id)
    result = observe_grounding_proposal(agent, proposal, probe.id, observer)
    assert not isinstance(result, Unknown), result
    assert len(observer.calls) == 1 and not devices.calls
    group = agent.interpretations.get(sg)
    assert group.selected_id == sc and group.candidates[0].payload.graph == graph
    candidate = next(c for c in group.candidates if c.id == result.scene_candidate_id)
    assert candidate.payload.graph.nodes == graph.nodes
    assert candidate.payload.graph.propositions[-1] == replace(probe.proposition, polarity=answer)
    assert agent.interpretations.get_source(proposal.evidence_source_id) == retained
    agent.interpretations.select(sg, candidate.id, reason='explicit observation scene choice')
    after = propose_scene_groundings(agent, model, lg, lc, PATH, sg, candidate.id)
    assert not isinstance(after, Unknown), after
    unknowns = [c.payload.reference for c in agent.interpretations.get(lg).candidates
                if c.id in after.unresolved_candidate_ids]
    assert None in unknowns
    if not answer:
        assert not after.candidate_ids and graph.nodes[0] in unknowns
        assert not devices.calls
        return
    assert graph.nodes[0] not in unknowns
    sentence, act, dependency = select_bound_request(agent, after, lg)
    def goal(frame):
        return GoalSpec((Condition('enabled', {'undergoer': frame.roles['object'].ref}),))
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda frame, *a, **kw:
        supplied_goal_batch(goal(frame), frame=frame))
    monkeypatch.setattr(devices, 'refine_goal', lambda lexical: goal(lexical.frame))
    outcome = agent.request(sentence, act, [], interpretation_dependency=dependency)
    assert outcome.status == 'done' and devices.calls == [graph.nodes[0]]
    assert agent.store.propositions() == []
