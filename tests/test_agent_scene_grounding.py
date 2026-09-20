"""Two learned bridges consume supplied scene graphs and exact mention structure.

The fixture adapter does not decode pixels; labels and executor effects are authored.
"""
from dataclasses import replace
import pytest
from tensorcode.agent import Agent, Plugin
from tensorcode.agent.core import InterpretationDecision
from tensorcode.agent.scene import SceneGraph, SceneProposal
from tensorcode.agent.scene_grounding import (retain_grounding_example, fit_grounding_model,
    admit_grounding_model, propose_scene_groundings, grounding_dependencies)
from tensorcode.agent.grounding import MentionBinding, propose_grounding
from tensorcode.agent.goal_learning import fit_goal_model, admit_goal_model
from tensorcode.agent.goal_interpretation import LearnedGoalProposal
from tensorcode.agent.task_dependencies import capture_dependency
from tensorcode.agent.understand import Act, Sentence, SentenceAlternative
from tensorcode.goals import Condition, GoalSpec
from tensorcode.language import Entity, Frame, Request, verbnet
from tensorcode.records import Proposition, Ref
from tensorcode.outcomes import Unknown
from agent_test_support import supplied_goal_batch, select_unique_fixture_goal
from test_agent_tasks import Devices

PATH = ('acts', 0, 'frame', 'roles', 'object')


class SuppliedScenes(Plugin):
    def __init__(self):
        super().__init__('authored-scene-fixture')
        self.active = 0
        self.double = False
        self.count = 0
    def interpret_image(self, image, image_ref):
        self.count += 1
        controls = tuple(Ref(f'device:scene-{self.count}-{n}') for n in range(2))
        panels = tuple(Ref(f'panel:scene-{self.count}-{n}') for n in range(2))
        facts = []
        for n in range(2):
            facts.extend((Proposition('caption', {'item': controls[n], 'value': 'same label'}),
                          Proposition('within', {'item': controls[n], 'container': panels[n]}),
                          Proposition('active', {'item': panels[n], 'value': self.double or n == self.active})))
        yield SceneProposal(SceneGraph(image_ref, (*controls, *panels), tuple(facts)),
                            provenance=('authored graph, no pixel decoding',))


def scene_and_mention(agent, scenes, *, active=0, double=False):
    scenes.active, scenes.double = active, double
    visual = agent.interpret_image(b'opaque original visual evidence')
    sg = agent.interpretations.get(visual.group_ids[0])
    sc = sg.candidates[0]
    agent.interpretations.select(sg.id, sc.id, reason='fixture explicitly admits this scene interpretation')
    frame = Frame('prepare', {'object': Entity('name', 'the active control')})
    act = Act('request', Request(frame), frame)
    source = agent.interpretations.add_source('prepare the active control', provider='authored meaning fixture')
    lg = agent.interpretations.create_group(source.id)
    candidate = agent.interpretations.propose(lg.id, SentenceAlternative(None, (act,)))
    return lg.id, candidate.id, sg.id, sc.id, sc.payload.graph


def fitted(agent, scenes):
    cases, teaching = [], []
    for active in (0, 1, 0):
        lg, lc, sg, sc, graph = scene_and_mention(agent, scenes, active=active)
        target = graph.nodes[active]
        negatives = tuple(r for r in graph.nodes if r != target)
        record = retain_grounding_example(agent, lg, lc, PATH, sg, sc,
            (target,), negatives, basis=('explicit teaching alignment and explicit exclusions',))
        assert not isinstance(record, Unknown), record
        teaching.append(record)
        cases.append((lg, lc, sg, sc, graph))
    handle = fit_grounding_model(agent, teaching[:2], teaching[2:], max_patterns=2048, max_matches=20000)
    assert not isinstance(handle, Unknown), handle
    handle = admit_grounding_model(agent, handle, reason='explicit model admission after independent scenes')
    assert not isinstance(handle, Unknown), handle
    return handle, cases


def select_bound_request(agent, report, lg):
    assert not isinstance(report, Unknown), report
    assert report.complete and len(report.candidate_ids) == 1
    candidate_id = report.candidate_ids[0]
    agent.interpretations.select(lg, candidate_id, reason='fixture explicitly chooses sole learned binding')
    candidate = next(c for c in agent.interpretations.get(lg).candidates if c.id == candidate_id)
    parent = capture_dependency(agent.interpretations, lg, basis=('explicit learned grounding choice',))
    act = candidate.payload.acts[0]
    sentence = Sentence('prepare the active control', (), None, (act,))
    return sentence, act, parent


def choose_learned_goal(group):
    proposals = [c for c in group.candidates if isinstance(c.payload, LearnedGoalProposal)]
    return InterpretationDecision(proposals[0].id if len(proposals) == 1 else None,
        'fixture explicitly chooses learned desired result', compared_revision=group.revision,
        compared_candidate_ids=tuple(c.id for c in group.candidates))


def test_learned_relational_grounding_then_learned_goal_executes_new_entity(monkeypatch):
    scenes, devices = SuppliedScenes(), Devices()
    agent = Agent([scenes, devices], goal_selector=select_unique_fixture_goal)
    model, cases = fitted(agent, scenes)
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda frame, *a, **kw:
        supplied_goal_batch(GoalSpec((Condition('enabled', {'undergoer': frame.roles['object'].ref}),)), frame=frame))
    monkeypatch.setattr(devices, 'refine_goal', lambda lexical:
        GoalSpec((Condition('enabled', {'undergoer': lexical.frame.roles['object'].ref}),)))
    teaching_tasks = []
    for lg, lc, sg, sc, graph in cases:
        report = propose_scene_groundings(agent, model, lg, lc, PATH, sg, sc)
        sentence, act, parent = select_bound_request(agent, report, lg)
        outcome = agent.request(sentence, act, [], interpretation_dependency=parent)
        assert outcome.status == 'done'
        teaching_tasks.append(outcome.task_id)
    goal_model = fit_goal_model(agent, teaching_tasks[:2], teaching_tasks[2:])
    assert not isinstance(goal_model, Unknown), goal_model
    goal_model = admit_goal_model(agent, goal_model, reason='explicit goal transfer admission')
    assert not isinstance(goal_model, Unknown), goal_model
    agent.goal_model = goal_model
    agent.goal_selector = choose_learned_goal
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda *a, **kw: pytest.fail('lexical goal fallback'))
    monkeypatch.setattr(devices, 'refine_goal', lambda *a: pytest.fail('authored goal mapping used'))
    lg, lc, sg, sc, graph = scene_and_mention(agent, scenes, active=1)
    report = propose_scene_groundings(agent, model, lg, lc, PATH, sg, sc)
    sentence, act, parent = select_bound_request(agent, report, lg)
    assert act.frame.roles['object'].ref == graph.nodes[1]
    outcome = agent.request(sentence, act, [], interpretation_dependency=parent)
    assert outcome.status == 'done' and devices.calls[-1] == graph.nodes[1]
    dependencies = agent.tasks.get(outcome.task_id).dependencies
    assert model.dependency in dependencies and goal_model.dependency in dependencies
    assert any(d.group_id == sg for d in dependencies)
    assert len(dependencies) == 5
    assert agent.store.propositions() == []


def test_two_matching_referents_remain_unselected_alternatives():
    scenes = SuppliedScenes()
    agent = Agent([scenes])
    model, _ = fitted(agent, scenes)
    lg, lc, sg, sc, graph = scene_and_mention(agent, scenes, double=True)
    report = propose_scene_groundings(agent, model, lg, lc, PATH, sg, sc)
    assert not isinstance(report, Unknown), report
    assert report.complete and len(report.candidate_ids) == 2
    group = agent.interpretations.get(lg)
    refs = {c.payload.acts[0].frame.roles['object'].ref for c in group.candidates if c.id in report.candidate_ids}
    assert refs == set(graph.nodes[:2]) and group.selected_id is None
    original = next(c for c in group.candidates if c.id == lc)
    assert original.payload.acts[0].frame.roles['object'].ref is None


@pytest.mark.parametrize('withdraw', ['scene', 'model'])
def test_withdrawn_grounding_support_blocks_before_goal_derivation(monkeypatch, withdraw):
    scenes, devices = SuppliedScenes(), Devices()
    agent = Agent([scenes, devices])
    model, _ = fitted(agent, scenes)
    lg, lc, sg, sc, _ = scene_and_mention(agent, scenes)
    report = propose_scene_groundings(agent, model, lg, lc, PATH, sg, sc)
    sentence, act, parent = select_bound_request(agent, report, lg)
    agent.interpretations.unset(sg if withdraw == 'scene' else model.group_id, reason='evidence reconsidered')
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda *a, **kw: pytest.fail('stale binding reached goal inference'))
    outcome = agent.request(sentence, act, [], interpretation_dependency=parent)
    assert outcome.status == 'unknown' and not devices.calls
    assert isinstance(outcome.verified, Unknown)


def test_successive_binding_keeps_learned_ancestry_and_withdrawal_guard(monkeypatch):
    scenes, devices = SuppliedScenes(), Devices()
    agent = Agent([scenes, devices])
    model, _ = fitted(agent, scenes)
    lg, lc, sg, sc, graph = scene_and_mention(agent, scenes)
    report = propose_scene_groundings(agent, model, lg, lc, PATH, sg, sc)
    assert not isinstance(report, Unknown)
    original = report.candidate_ids[0]
    child = propose_grounding(agent.interpretations, lg, original, (
        MentionBinding(PATH, graph.nodes[0], (report.source_id,), 'compatible subsequent binding'),))
    dependencies = grounding_dependencies(agent, lg, child.id)
    assert dependencies == grounding_dependencies(agent, lg, original)
    assert model.dependency in dependencies and any(d.group_id == sg for d in dependencies)
    sentence, act, parent = select_bound_request(agent, replace(report, candidate_ids=(child.id,)), lg)
    agent.interpretations.unset(model.group_id, reason='withdraw the ancestor model')
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda *a, **kw: pytest.fail('lost ancestor dependency'))
    outcome = agent.request(sentence, act, [], interpretation_dependency=parent)
    assert outcome.status == 'unknown' and not devices.calls


def test_turn_blocks_selected_grounding_with_withdrawn_scene(monkeypatch):
    from types import SimpleNamespace
    from tensorcode.agent.operations import Transcript
    scenes, devices = SuppliedScenes(), Devices()
    agent = Agent([scenes, devices])
    model, _ = fitted(agent, scenes)
    lg, lc, sg, sc, _ = scene_and_mention(agent, scenes)
    report = propose_scene_groundings(agent, model, lg, lc, PATH, sg, sc)
    sentence, _, _ = select_bound_request(agent, report, lg)
    monkeypatch.setattr(agent, 'interpret', lambda text: SimpleNamespace(
        transcript=Transcript((sentence,), 'retained supplied reading'), group_ids=(lg,), unavailable=None))
    def select(group):
        return InterpretationDecision(report.candidate_ids[0], 'explicit retained reading selection',
            compared_revision=group.revision, compared_candidate_ids=tuple(c.id for c in group.candidates))
    agent.interpretation_selector = select
    agent.interpretations.unset(sg, reason='withdraw visual interpretation before turn')
    monkeypatch.setattr(agent, 'handle', lambda *a, **kw: pytest.fail('stale scene reached dispatch'))
    turn = agent.turn('prepare the active control')
    assert turn.outcomes[0].status == 'unknown' and not devices.calls
    assert isinstance(turn.outcomes[0].verified, Unknown)
