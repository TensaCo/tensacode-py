"""Authored lexical alternatives exercise goal selection through actual requests."""
import pytest

from tensorcode.agent import Agent, InterpretationDecision
from tensorcode.agent.operations import Transcript
from tensorcode.agent.understand import Act, Sentence
from tensorcode.language import Frame, Request, verbnet
from tensorcode.outcomes import Unknown
from tensorcode.records import Ref
from agent_test_support import select_fixture_reading
from test_agent_tasks import Devices, desired


def resource():
    syntax = (('NP', 'Agent'), ('VERB', ''), ('NP', 'Theme'))
    return {'prepare': tuple(verbnet.VerbClass(name, ('prepare',), (verbnet.VFrame('NP V NP', syntax, (
        verbnet.Pred('exists', (('Event', 'e1'), ('ThemRole', 'Theme'))),
        verbnet.Pred(predicate, (('Event', 'e2'), ('ThemRole', 'Theme'))),
    )),)) for name, predicate in [('authored:enable', 'enabled'), ('authored:other', 'other')])}


def choose_enabled(group):
    choices = [c for c in group.candidates if isinstance(c.payload, verbnet.GoalProposal)
               and c.payload.goal.conditions[0].pred == 'enabled']
    assert len(choices) == 1
    return InterpretationDecision(choices[0].id, 'Fixture explicitly supplies intended enabled state')


def setup(monkeypatch, *, selector=None, budget=256):
    from tensorcode.agent import core
    frame = Frame('prepare', {'object': Ref('device:a')}, {'mood': 'imperative'})
    sentence = Sentence('prepare a', ('prepare', 'a'), None,
                        (Act('request', Request(frame), frame),))
    monkeypatch.setattr(core.ops, 'parse', lambda *a, **kw: Transcript((sentence,), 'authored input fixture'))
    plugin = Devices()
    agent = Agent([plugin], interpretation_selector=select_fixture_reading,
                  goal_selector=selector, goal_derivation_budget=budget)
    agent.verbs = resource()
    return agent, plugin


@pytest.mark.parametrize('singleton', [False, True])
def test_selected_sentence_does_not_select_its_lexical_goal(monkeypatch, singleton):
    agent, plugin = setup(monkeypatch)
    if singleton:
        agent.verbs['prepare'] = agent.verbs['prepare'][:1]
    outcome = agent.turn('prepare a').outcomes[0]
    assert outcome.status == 'unknown' and not plugin.calls
    group = agent.interpretations.get(outcome.goal_interpretation_id)
    assert group.selected_id is None and len(group.candidates) == (1 if singleton else 2)
    assert {c.payload.goal.conditions[0].pred for c in group.candidates} == ({'enabled'} if singleton else {'enabled', 'other'})
    source = agent.interpretations.get_source(group.source_id)
    assert source.text == 'prepare a'
    task = agent.tasks.get(outcome.task_id)
    assert len(task.dependencies) == 1
    assert source.metadata['parent_dependency'] == task.dependencies[0]
    assert not agent.store.propositions()


def test_explicit_goal_selection_executes_with_both_interpretation_dependencies(monkeypatch):
    agent, plugin = setup(monkeypatch, selector=choose_enabled)
    outcome = agent.turn('prepare a').outcomes[0]
    assert outcome.status == 'done' and plugin.calls == [Ref('device:a')]
    task = agent.tasks.get(outcome.task_id)
    assert len(task.dependencies) == 2
    assert [dep.group_id for dep in task.dependencies] == [outcome.interpretation_id, outcome.goal_interpretation_id]
    assert task.revisions[0].dependencies == task.dependencies
    assert task.attempts[0].receipt == outcome.receipt


@pytest.mark.parametrize('stage', ['precondition', 'execute'])
def test_changed_goal_interpretation_stops_dispatch_or_completion(monkeypatch, stage):
    agent, plugin = setup(monkeypatch, selector=choose_enabled)
    def withdraw():
        groups = [g for g in agent.interpretations.values()
                  if g.candidates and isinstance(g.candidates[0].payload, verbnet.GoalProposal)]
        assert len(groups) == 1
        agent.interpretations.unset(groups[0].id, reason='Counterevidence against goal interpretation')
    if stage == 'precondition':
        original = plugin.precondition_holds
        def changed(*args):
            withdraw()
            return original(*args)
        plugin.precondition_holds = changed
    else:
        original = plugin.execute
        def changed(*args, **kwargs):
            receipt = original(*args, **kwargs)
            withdraw()
            return receipt
        plugin.execute = changed
    outcome = agent.turn('prepare a').outcomes[0]
    assert outcome.status == 'unknown'
    assert outcome.verified.reason == 'interpretation_dependency_changed'
    assert len(plugin.calls) == (0 if stage == 'precondition' else 1)
    assert outcome.receipt.status == ('rejected' if stage == 'precondition' else 'applied')
    assert len(agent.tasks.get(outcome.task_id).dependencies) == 2


def test_goal_search_budget_cannot_authorize_visible_prefix(monkeypatch):
    def forbidden(group):
        pytest.fail('incomplete goal search reached selector')
    agent, plugin = setup(monkeypatch, selector=forbidden, budget=0)
    outcome = agent.turn('prepare a').outcomes[0]
    assert outcome.status == 'unknown' and not plugin.calls
    assert outcome.goal_interpretation_id
    assert isinstance(outcome.goal, Unknown)


def test_structured_goals_do_not_enter_lexical_search(monkeypatch):
    monkeypatch.setattr(verbnet, 'goal_candidates', lambda *a, **kw: pytest.fail('structured goal entered lexical search'))
    plugin = Devices()
    outcome = Agent([plugin]).pursue(desired())
    assert outcome.status == 'done'
