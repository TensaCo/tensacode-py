"""Authored alternatives isolate continuation/dispatch, not language accuracy."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tensorcode import ops
from tensorcode.agent import Agent, InterpretationDecision
from tensorcode.agent.operations import Transcript
from tensorcode.agent.understand import SentenceAlternative
from interpretation_fixtures import project_sentence


class SuppliedContinuation:
    def __init__(self, alternative):
        self.alternative = alternative
        self.pending = 1

    def advance(self, *, max_expansions, max_candidates):
        if self.pending and max_expansions and max_candidates:
            self.pending = 0
            return SimpleNamespace(alternatives=(self.alternative,), explored=1, pending=0)
        return SimpleNamespace(alternatives=(), explored=0, pending=self.pending)


def source(monkeypatch, *, two=False):
    first, second = (project_sentence(name) for name in ('hello', 'demo'))
    alternative = SentenceAlternative(second.reading, second.acts, provenance='supplied')
    first = replace(first, continuation=SuppliedContinuation(alternative))
    monkeypatch.setattr(ops, 'parse', lambda *a, **kw: Transcript((first, second) if two else (first,), 'supplied'))
    return first


def test_expand_old_group_after_another_turn_without_reparsing(monkeypatch):
    sentence = source(monkeypatch)
    agent = Agent([])
    initial = agent.interpret('ambiguous source')
    old = agent.interpretations.get(initial.group_ids[0])
    agent.turn('intervening turn')
    monkeypatch.setattr(ops, 'parse', lambda *a, **kw: pytest.fail('must not reparse'))
    result = agent.expand_interpretation(old.id, max_expansions=1, max_candidates=1)
    group = agent.interpretations.get(old.id)
    assert group.source_id == old.source_id
    assert group.candidates[:len(old.candidates)] == old.candidates
    assert len(group.candidates) == len(old.candidates) + 1
    assert result.explored == 1 and result.pending == 0
    assert group.selected_id is None
    assert sentence.continuation.pending == 1
    assert len(agent.turns) == 1
    again = agent.expand_interpretation(old.id, max_expansions=1, max_candidates=1)
    assert again.candidate_ids == ()


def test_selector_cannot_commit_after_expanding_its_comparison_set(monkeypatch):
    source(monkeypatch)
    agent = Agent([])
    group_id, = agent.interpret('ambiguous').group_ids
    def select(group):
        agent.expand_interpretation(group.id, max_expansions=1, max_candidates=1)
        return InterpretationDecision(group.candidates[0].id, 'stale supplied choice')
    agent.interpretation_selector = select
    with pytest.raises(RuntimeError, match='changed during selection'):
        agent._select_interpretation(group_id)
    assert agent.interpretations.get(group_id).selected_id is None


def test_later_sentence_cannot_leave_earlier_stale_acts_executable(monkeypatch):
    source(monkeypatch, two=True)
    agent = Agent([])
    seen = []
    def select(group):
        if seen:
            agent.expand_interpretation(seen[0], max_expansions=1, max_candidates=1)
        seen.append(group.id)
        return InterpretationDecision(group.candidates[0].id, 'explicit supplied choice')
    agent.interpretation_selector = select
    handled = []
    original = agent.handle
    def handle(sentence, act, *args, **kwargs):
        handled.append(sentence.text)
        return original(sentence, act, *args, **kwargs)
    monkeypatch.setattr(agent, 'handle', handle)
    turn = agent.turn('two supplied sentences')
    assert any(event['type'] == 'interpretation_stale' for event in turn.events)
    assert turn.sentences[0].text not in handled
    assert agent.interpretations.get(seen[0]).selected_id is None


def test_expanded_meaning_can_win_fresh_investigation_without_execution(tmp_path, monkeypatch):
    from tensorcode.agent import CandidateHypothesis, Condition, FileSystemPlugin
    source(monkeypatch)
    agent = Agent([FileSystemPlugin(tmp_path)])
    interpreted = agent.interpret('supplied ambiguity')
    group = agent.interpretations.get(interpreted.group_ids[0])
    assert len(group.candidates) == 1
    agent.interpretations.select(group.id, group.candidates[0].id, reason='earlier supplied policy')
    result = agent.expand_interpretation(group.id, max_expansions=1, max_candidates=1)
    assert result.group.selected_id is None
    (tmp_path / 'current').write_text('demo')
    hypotheses = tuple(CandidateHypothesis(candidate.id,
        (Condition('content', {'path': 'current', 'text': name}),), ('authored meaning prediction',))
        for candidate, name in zip(result.group.candidates, ('hello', 'demo')))
    with pytest.raises(ValueError, match='every non-rejected'):
        agent.investigate_interpretation(group.id, hypotheses[:1])
    investigation = agent.investigate_interpretation(group.id, hypotheses)
    assert investigation.decision.candidate_id == result.candidate_ids[0]
    assert not (tmp_path / 'demo').exists()
    assert agent.turns == []
    assert agent.interpretations.get_source(interpreted.source_id).text == 'supplied ambiguity'


def test_interpretation_is_rechecked_between_acts(monkeypatch):
    from tensorcode.agent.core import Outcome
    sentence = source(monkeypatch)
    alternative = SentenceAlternative(sentence.reading, sentence.acts * 2, provenance='supplied compound')
    # Requests now require no known pending work before their first dispatch.
    # Introduce new search work during that first act to test the inter-act guard.
    sentence = replace(sentence, acts=alternative.acts, alternatives=(alternative,), continuation=None)
    monkeypatch.setattr(ops, 'parse', lambda *a, **kw: Transcript((sentence,), 'supplied'))
    agent = Agent([], interpretation_selector=lambda group: InterpretationDecision(group.candidates[0].id, 'supplied'))
    handled = []
    def handle(s, act, *args, **kwargs):
        handled.append(act)
        group, = agent.interpretations.values()
        agent.interpretations.attach_continuation(group.id, SuppliedContinuation(alternative))
        agent.expand_interpretation(group.id, max_expansions=1, max_candidates=1)
        return Outcome(act, 'noted')
    monkeypatch.setattr(agent, 'handle', handle)
    turn = agent.turn('supplied compound')
    assert len(handled) == 1
    assert [o.status for o in turn.outcomes] == ['noted', 'unknown']


def test_selected_request_with_pending_search_does_not_dispatch(monkeypatch):
    source(monkeypatch)
    agent = Agent([], interpretation_selector=lambda group: InterpretationDecision(group.candidates[0].id, 'supplied'))
    monkeypatch.setattr(agent, 'handle', lambda *a, **kw: pytest.fail('pending request must not dispatch'))
    turn = agent.turn('supplied request with unfinished alternatives')
    assert turn.outcomes[0].status == 'unknown'
    assert 'pending' in turn.outcomes[0].reason
    assert not agent.tasks.values()


def test_selector_can_explicitly_acknowledge_expanded_comparison_set(monkeypatch):
    source(monkeypatch)
    agent = Agent([])
    group_id, = agent.interpret('ambiguous').group_ids
    def select(group):
        result = agent.expand_interpretation(group.id, max_expansions=1, max_candidates=1)
        current = result.group
        return InterpretationDecision(result.candidate_ids[0], 'explicit policy over expanded set',
            compared_revision=current.revision,
            compared_candidate_ids=tuple(c.id for c in current.candidates))
    agent.interpretation_selector = select
    decision = agent._select_interpretation(group_id)
    assert agent.interpretations.get(group_id).selected_id == decision.candidate_id


def test_partial_comparison_acknowledgement_is_rejected(monkeypatch):
    source(monkeypatch)
    agent = Agent([])
    group_id, = agent.interpret('ambiguous').group_ids
    def select(group):
        result = agent.expand_interpretation(group.id, max_expansions=1, max_candidates=1)
        return InterpretationDecision(result.candidate_ids[0], 'omits earlier rival',
            compared_revision=result.group.revision, compared_candidate_ids=result.candidate_ids)
    agent.interpretation_selector = select
    with pytest.raises(RuntimeError, match='exact comparison basis'):
        agent._select_interpretation(group_id)
