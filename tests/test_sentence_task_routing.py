"""Authored reading/goal fixtures isolate the missing discourse decision.

These tests do not claim learned language or inferred goals. A selected request
and a selected goal still do not establish whether to start or revise a task.
"""
import pytest

from tensorcode.agent import Agent
from tensorcode.agent.operations import Transcript
from tensorcode.agent.understand import Act, Sentence
from tensorcode.goals import Condition, GoalSpec
from tensorcode.language import Frame, Request
from tensorcode.records import Ref

from agent_test_support import select_fixture_reading
from test_agent_goal_interpretations import resource, choose_enabled
from test_agent_tasks import Devices


@pytest.mark.parametrize('act_count', [1, 2])
@pytest.mark.parametrize('existing_task', [False, True])
def test_missing_discourse_evidence_neither_creates_tasks_nor_dispatches(
        monkeypatch, act_count, existing_task):
    from tensorcode.agent import core

    frames = tuple(Frame('prepare', {'object': Ref(f'device:{index}')},
                         {'mood': 'imperative'}) for index in range(act_count))
    sentence = Sentence('supplied complete request', (), None,
                        tuple(Act('request', Request(frame), frame) for frame in frames))
    monkeypatch.setattr(core.ops, 'parse',
                        lambda *a, **kw: Transcript((sentence,), 'authored routing fixture'))
    plugin = Devices()
    agent = Agent([plugin], interpretation_selector=select_fixture_reading,
                  goal_selector=choose_enabled)
    agent.verbs = resource()
    if existing_task:
        agent.tasks.create('supplied earlier task', GoalSpec((
            Condition('enabled', {'item': Ref('device:earlier')}),)))
    before = agent.tasks.values()

    result = agent.turn(sentence.text)

    assert agent.tasks.values() == before, 'Unresolved discourse created or changed tasks'
    assert plugin.calls == [], 'Selected syntax/goal bypassed the discourse decision'
    assert result.outcomes and all(outcome.status == 'unknown' for outcome in result.outcomes)
    assert all(outcome.task_id is None for outcome in result.outcomes)
