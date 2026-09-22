"""Public factories preserve mechanism behavior; random weights imply no quality."""
import importlib.util

import pytest

from tensorcode.tools import Investigator, Planner


def test_runtime_namespace_is_removed():
    assert importlib.util.find_spec('tensorcode.runtime') is None


def test_cognitive_session_public_lifecycle(tmp_path):
    from tensorcode.tools.cognition import Evidence
    from tensorcode.tools.investigator import Evidence as InvestigationEvidence, InvestigationSession
    assert Evidence is InvestigationEvidence
    tool = Investigator({'vocabulary': ['alpha', 'beta'], 'dimensions': 8, 'slots': 2, 'steps': 1})
    session = tool.new_cognitive_session(policy={'min_support': .8},
                                        memory={'capacity': 8, 'top_k': 2}, max_records=8)
    assert type(session) is InvestigationSession
    other = tool.new_cognitive_session(memory={'capacity': 8})
    session.ingest([Evidence('a', 'alpha', 'source')])
    session.remember('a')
    session.revise_evidence('a', 'beta')
    hits = session.retrieve('beta')
    assert hits[0].evidence.text == 'beta'
    assert hits[0].evidence.source_id == 'source'
    assert other.active_evidence == ()
    assert other.retrieve('beta') == ()
    path = tmp_path / 'session.json'
    session.save(path)
    restored = tool.load_cognitive_session(path)
    assert restored.snapshot() == session.snapshot()
    assert restored.investigator is tool
    restored.remove_evidence('a')
    assert session.active_evidence
    assert not restored.active_evidence
    assert tool.new_session().history == []


@pytest.mark.parametrize('options', [
    {'policy': lambda x: x}, {'memory': object()}, {'memory': {'obsolete': True}},
    {'policy': {'obsolete': True}}, {'max_records': 0},
])
def test_cognitive_factory_rejects_non_json_or_obsolete_options(options):
    tool = Investigator({'vocabulary': ['alpha'], 'dimensions': 8, 'slots': 2, 'steps': 1})
    with pytest.raises((TypeError, ValueError)):
        tool.new_cognitive_session(**options)


def test_public_plan_factory_validates_before_effects_and_bounds_receipts():
    from tensorcode.tools.actions import ActionOutcome
    from tensorcode.tools.planner import ExecutablePlan, PlanStep
    tool = Planner({'vocabulary': ['alpha'], 'dimensions': 8, 'slots': 2, 'steps': 1})
    calls = []
    def act(state):
        calls.append(state)
        return ActionOutcome(state + 1, {'count': state + 1})
    plan = ExecutablePlan('chosen', (PlanStep('act'),))
    executor = tool.new_executor(actions={'act': act}, replan=lambda request: plan, max_steps=2)
    assert calls == []
    with pytest.raises(ValueError):
        executor(0, ExecutablePlan('invalid', (PlanStep('act'), PlanStep('missing'))))
    assert calls == []
    result = executor(0, plan)
    assert result.state == 2
    assert len(result.experiences) == 2
    assert result.stop_reason == 'budget_exhausted'


def test_public_action_loop_factory_constructs_without_running_callbacks():
    from tensorcode.tools.actions import ActionOutcome, ActionRequest, action_loop
    calls = []
    def choose(request, *, context=None):
        assert isinstance(request, ActionRequest)
        return 'increment'
    def increment(state):
        calls.append(state)
        return ActionOutcome(state + 1, {'previous': state})
    loop = action_loop(chooser=choose, actions={'increment': increment}, max_steps=2)
    assert calls == []
    result = loop(0)
    assert result.state == 2
    assert [receipt.effect for receipt in result.receipts] == [{'previous': 0}, {'previous': 1}]


def test_public_cognitive_investigation_preserves_source_evidence(tmp_path):
    from tensorcode.tools.cognition import Evidence
    from test_investigation import config
    tool = Investigator(config()).eval()
    session = tool.new_cognitive_session(memory={'capacity': 8}, max_records=16)
    session.ingest([Evidence('original', 'hello', 'document')])
    first = session.investigate('hello', hypotheses=[{'id': 'candidate', 'text': 'world'}])
    assert first['evidence'] == [{'id': 'original', 'text': 'hello', 'source_id': 'document'}]
    assert session.state.hypotheses[0].text == 'world'
    assert not session.state.observations
    session.remember('original')
    session.revise_evidence('original', 'world')
    second = session.investigate('hello', hypotheses=[{'id': 'candidate', 'text': 'world'}])
    assert second['evidence'][0]['text'] == 'world'
    assert second['evidence'][0]['source_id'] == 'document'
    path = tmp_path / 'cognitive.json'
    session.save(path)
    assert tool.load_cognitive_session(path).snapshot() == session.snapshot()
