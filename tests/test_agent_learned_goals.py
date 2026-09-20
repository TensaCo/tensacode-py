"""Learned transfer from explicit teaching tasks; initial grounding is supplied."""
import pytest

from tensorcode.agent import Agent
from tensorcode.agent.core import InterpretationDecision
from tensorcode.agent.goal_interpretation import LearnedGoalProposal
from tensorcode.agent.goal_learning import fit_goal_model, admit_goal_model
from tensorcode.agent.understand import Act, Sentence
from tensorcode.agent.task_dependencies import capture_dependency
from tensorcode.goals import GoalSpec
from tensorcode.language import Frame, Request, verbnet
from tensorcode.outcomes import Unknown
from tensorcode.records import Ref
from agent_test_support import supplied_goal_batch, select_unique_fixture_goal
from test_agent_tasks import Devices, desired


def request(agent, name, *, features=None):
    frame = Frame('prepare', {'object': Ref('device:' + name)}, features or {})
    act = Act('request', Request(frame), frame)
    sentence = Sentence('prepare ' + name, (), None, (act,))
    source = agent.interpretations.add_source(sentence.text, provider='explicit teaching fixture')
    group = agent.interpretations.create_group(source.id)
    candidate = agent.interpretations.propose(group.id, sentence)
    agent.interpretations.select(group.id, candidate.id, reason='fixture supplies grounded source meaning')
    parent = capture_dependency(agent.interpretations, group.id, basis=('authored grounded input',))
    return agent.request(sentence, act, [], interpretation_dependency=parent)


def trained(monkeypatch):
    plugin = Devices()
    agent = Agent([plugin], goal_selector=select_unique_fixture_goal)
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda frame, *a, **kw:
        supplied_goal_batch(GoalSpec(desired().conditions), frame=frame))
    # Explicit teacher mapping is used to obtain training labels only.
    monkeypatch.setattr(plugin, 'refine_goal', lambda lexical:
        desired(lexical.frame.roles['object'].id.split(':', 1)[1]))
    tasks = [request(agent, name).task_id for name in ('a', 'b', 'heldout')]
    fitted = fit_goal_model(agent, tasks[:2], tasks[2:])
    assert not isinstance(fitted, Unknown), fitted
    admitted = admit_goal_model(agent, fitted, reason='explicitly admit held-out tested structural transfer')
    assert not isinstance(admitted, Unknown), admitted
    agent.goal_model = admitted
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda *a, **kw: pytest.fail('learned path fell back to lexical projection'))
    monkeypatch.setattr(plugin, 'refine_goal', lambda *a: pytest.fail('prediction used authored refiner'))
    return agent, plugin, tasks, admitted


def choose_learned(group):
    proposals = [c for c in group.candidates if isinstance(c.payload, LearnedGoalProposal)]
    return InterpretationDecision(proposals[0].id if len(proposals) == 1 else None,
        'test explicitly authorizes sole learned goal', compared_revision=group.revision,
        compared_candidate_ids=tuple(c.id for c in group.candidates))


def test_new_entity_goal_executes_without_lexical_projection_or_authored_refiner(monkeypatch):
    agent, plugin, tasks, model = trained(monkeypatch)
    agent.goal_selector = choose_learned
    result = request(agent, 'new')
    assert result.status == 'done'
    assert result.goal.conditions == desired('new').conditions
    assert plugin.calls[-1] == Ref('device:new')
    task = agent.tasks.get(result.task_id)
    assert model.dependency in task.dependencies
    group = agent.interpretations.get(result.goal_interpretation_id)
    assert isinstance(group.selected.payload, LearnedGoalProposal)
    assert set(group.selected.payload.training_example_ids)
    source = agent.interpretations.get_source(group.source_id)
    assert source.provider == 'learned-goal-correspondence'
    assert source.payload['frame'].roles['object'] == Ref('device:new')


def test_deferred_learned_goal_can_be_adopted_and_pursued(monkeypatch):
    agent, plugin, _, model = trained(monkeypatch)
    agent.goal_selector = None
    result = request(agent, 'deferred')
    assert result.status == 'unknown'
    before = list(plugin.calls)
    group = agent.interpretations.get(result.goal_interpretation_id)
    adopted = agent.adopt_task_goal(result.task_id, choose_learned(group), reason='explicit later goal choice')
    assert not isinstance(adopted, Unknown), adopted
    assert model.dependency in adopted.dependencies and plugin.calls == before
    assert agent.pursue(task_id=adopted.id).status == 'done'
    assert plugin.calls[-1] == Ref('device:deferred')


def test_unknown_qualifier_and_missing_model_never_fall_back(monkeypatch):
    agent, plugin, _, model = trained(monkeypatch)
    agent.goal_selector = choose_learned
    before = list(plugin.calls)
    result = request(agent, 'new', features={'negated': True})
    assert result.status == 'unknown' and plugin.calls == before
    agent.interpretations.unset(model.group_id, reason='model withdrawn')
    result = request(agent, 'other')
    assert result.status == 'unknown' and plugin.calls == before


def test_model_refit_invalidates_already_adopted_task(monkeypatch):
    agent, plugin, tasks, model = trained(monkeypatch)
    agent.goal_selector = None
    outcome = request(agent, 'waiting')
    adopted = agent.adopt_task_goal(outcome.task_id,
        choose_learned(agent.interpretations.get(outcome.goal_interpretation_id)), reason='explicit goal choice')
    assert not isinstance(adopted, Unknown), adopted
    before = list(plugin.calls)
    replacement = fit_goal_model(agent, tasks[:2], tasks[2:], group_id=model.group_id)
    assert not isinstance(replacement, Unknown), replacement
    result = agent.pursue(task_id=adopted.id)
    assert result.status == 'unknown' and plugin.calls == before
    assert result.verified.reason == 'interpretation_dependency_changed'


def test_withdrawing_model_inside_selector_cannot_authorize_dispatch(monkeypatch):
    agent, plugin, _, model = trained(monkeypatch)
    before = list(plugin.calls)
    def withdraw(group):
        agent.interpretations.unset(model.group_id, reason='evidence policy changed during choice')
        return choose_learned(group)
    agent.goal_selector = withdraw
    result = request(agent, 'new')
    assert result.status == 'unknown' and plugin.calls == before


def test_withdrawing_model_during_precondition_blocks_external_call(monkeypatch):
    agent, plugin, _, model = trained(monkeypatch)
    agent.goal_selector = choose_learned
    before = list(plugin.calls)
    def precondition(condition, args):
        agent.interpretations.unset(model.group_id, reason='fresh evidence withdrew model')
        return True
    monkeypatch.setattr(plugin, 'precondition_holds', precondition)
    result = request(agent, 'new')
    assert result.status == 'unknown' and plugin.calls == before
    assert result.verified.reason == 'interpretation_dependency_changed'


def test_learned_goal_still_depends_on_original_reading(monkeypatch):
    agent, plugin, _, model = trained(monkeypatch)
    agent.goal_selector = None
    outcome = request(agent, 'later')
    task = agent.tasks.get(outcome.task_id)
    parent = task.dependencies[0]
    adopted = agent.adopt_task_goal(task.id,
        choose_learned(agent.interpretations.get(outcome.goal_interpretation_id)), reason='explicit choice')
    assert len(adopted.dependencies) == 3 and parent in adopted.dependencies
    before = list(plugin.calls)
    agent.interpretations.unset(parent.group_id, reason='source reading corrected')
    result = agent.pursue(task_id=task.id)
    assert result.status == 'unknown' and plugin.calls == before
