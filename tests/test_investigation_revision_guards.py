"""Callback mutation cannot turn stale supplied predictions into authority."""
from types import SimpleNamespace

import pytest

from tensorcode import ops
from tensorcode.agent import Agent, CandidateHypothesis, Condition, InterpretationDecision
from tensorcode.agent.operations import Transcript
from tensorcode.agent.plugin import Plugin
from tensorcode.agent.understand import SentenceAlternative
from interpretation_fixtures import project_sentence


class Pending:
    pending = 1

    def advance(self, *, max_expansions, max_candidates):
        return SimpleNamespace(alternatives=(), explored=0, pending=self.pending)


def setup():
    provider = Plugin('observer')
    provider.observe_condition = lambda condition: True
    agent = Agent([provider])
    source = agent.interpretations.add_source('authored evidence')
    group = agent.interpretations.create_group(source.id)
    candidate = agent.interpretations.propose(group.id, SentenceAlternative(None, ()))
    return agent, provider, group.id, candidate.id


def hypotheses(group):
    return tuple(CandidateHypothesis(c.id, (Condition('fixture', {}),), ('authored prediction',))
                 for c in group.candidates if not c.rejected)


@pytest.mark.parametrize('mutation', ['select', 'attach', 'propose'])
def test_producer_mutation_invalidates_predicted_comparison_before_observation(mutation):
    agent, provider, group_id, candidate_id = setup()
    calls = []
    provider.observe_condition = lambda condition: calls.append(condition) or True
    def produce(group):
        old = hypotheses(group)
        if mutation == 'select':
            agent.interpretations.select(group.id, candidate_id, reason='callback selection')
        elif mutation == 'attach':
            agent.interpretations.attach_continuation(group.id, Pending())
        else:
            agent.interpretations.propose(group.id, SentenceAlternative(None, ()))
        return old
    with pytest.raises(RuntimeError, match='during hypothesis generation'):
        agent.resolve_interpretation(group_id, produce)
    assert not calls
    assert len(agent.interpretations.sources()) == 1


@pytest.mark.parametrize('mutation', ['select', 'attach', 'propose'])
def test_observer_mutation_invalidates_observations_even_when_ids_unchanged(mutation):
    agent, provider, group_id, candidate_id = setup()
    def observe(condition):
        if mutation == 'select':
            agent.interpretations.select(group_id, candidate_id, reason='callback selection')
        elif mutation == 'attach':
            agent.interpretations.attach_continuation(group_id, Pending())
        else:
            agent.interpretations.propose(group_id, SentenceAlternative(None, ()))
        return True
    provider.observe_condition = observe
    with pytest.raises(RuntimeError, match='during investigation'):
        agent.investigate_interpretation(group_id, hypotheses(agent.interpretations.get(group_id)))
    assert len(agent.interpretations.sources()) == 1


@pytest.mark.parametrize('parameter', ['max_expansions', 'max_candidates', 'max_probes'])
@pytest.mark.parametrize('value', [True, -1, 1.2])
def test_invalid_resolution_budgets_have_no_side_effects(parameter, value):
    agent, _, group_id, _ = setup()
    agent.interpretations.attach_continuation(group_id, Pending())
    before = agent.interpretations.get(group_id)
    status = agent.interpretations.continuation_status(group_id)
    with pytest.raises(ValueError):
        agent.resolve_interpretation(group_id, hypotheses, **{parameter: value})
    assert agent.interpretations.get(group_id) == before
    assert agent.interpretations.continuation_status(group_id) == status


def test_later_policy_attaching_pending_work_invalidates_earlier_dispatch(monkeypatch):
    first, second = (project_sentence(name) for name in ('hello', 'demo'))
    monkeypatch.setattr(ops, 'parse', lambda *a, **kw: Transcript((first, second), 'supplied'))
    agent = Agent([])
    seen = []
    def select(group):
        if seen:
            agent.interpretations.attach_continuation(seen[0], Pending())
        seen.append(group.id)
        return InterpretationDecision(group.candidates[0].id, 'explicit supplied policy')
    agent.interpretation_selector = select
    handled = []
    original = agent.handle
    def handle(sentence, act, *args, **kwargs):
        handled.append(sentence.text)
        return original(sentence, act, *args, **kwargs)
    monkeypatch.setattr(agent, 'handle', handle)
    turn = agent.turn('two authored sentences')
    assert first.text not in handled
    assert any(e['type'] == 'interpretation_stale' for e in turn.events)
