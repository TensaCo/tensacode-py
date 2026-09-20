"""Authored meanings/predictions isolate pending-search investigation policy."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorcode import ops
from tensorcode.agent import Agent, CandidateHypothesis, Condition, FileSystemPlugin, RefinementLibrary
from tensorcode.agent.operations import Transcript
from tensorcode.agent.understand import SentenceAlternative
from interpretation_fixtures import project_sentence
from agent_test_support import fixture_goal_selector


class SuppliedPending:
    def __init__(self, alternatives):
        self.alternatives = list(alternatives)

    @property
    def pending(self):
        return len(self.alternatives)

    def advance(self, *, max_expansions, max_candidates):
        count = min(max_expansions, max_candidates, self.pending)
        emitted = tuple(self.alternatives[:count])
        del self.alternatives[:count]
        return SimpleNamespace(alternatives=emitted, explored=count, pending=self.pending)


def setup_source(monkeypatch, *, pending=1):
    initial, later = (project_sentence(name) for name in ('hello', 'demo'))
    proposal = SentenceAlternative(later.reading, later.acts, provenance='test:authored later meaning')
    sentence = replace(initial, continuation=SuppliedPending([proposal] * pending))
    monkeypatch.setattr(ops, 'parse', lambda *args, **kwargs: Transcript((sentence,), 'test:authored source'))


def supplied_models(group):
    return tuple(CandidateHypothesis(c.id,
        (Condition('content', {'path': 'current', 'text': 'demo'}, negated=index == 0),),
        ('test authors world-conditional prediction; not inferred intent',))
        for index, c in enumerate(group.candidates))


def make_agent(path, **kwargs):
    library = RefinementLibrary.load(Path(__file__).parent / 'fixtures/project_refinements.json')
    return Agent([FileSystemPlugin(path, refinements=library)], **kwargs)


def test_direct_visible_winner_is_withheld_while_unseen_rival_pending(tmp_path, monkeypatch):
    setup_source(monkeypatch)
    (tmp_path / 'current').write_text('hello')
    agent = make_agent(tmp_path)
    group_id, = agent.interpret('authored ambiguity').group_ids
    group = agent.interpretations.get(group_id)
    result = agent.investigate_interpretation(group_id, supplied_models(group), max_probes=1)
    assert result.result.selected_id is None and result.decision.candidate_id is None
    assert result.result.reason == 'interpretation_search_pending'
    evidence = agent.interpretations.get_source(result.evidence_source_id)
    assert evidence.payload['candidate_result'].selected_id == group.candidates[0].id
    assert evidence.metadata['continuation_status'].pending == 1
    assert agent.interpretations.get(group_id).selected_id is None
    assert len(agent.tasks) == 0


def test_automatic_expansion_exposes_new_winner_before_supplied_investigation(tmp_path, monkeypatch):
    setup_source(monkeypatch)
    (tmp_path / 'current').write_text('demo')
    compared = []
    def producer(group):
        compared.append(tuple(c.id for c in group.candidates))
        return supplied_models(group)
    agent = make_agent(tmp_path, interpretation_hypotheses=producer,
                       interpretation_expansion_budget=1, interpretation_candidate_budget=1,
                       interpretation_probe_budget=1,
                       goal_selector=fixture_goal_selector('build-26.1-1', frame_index=0))
    turn = agent.turn('authored ambiguity')
    group = agent.interpretations.get(turn.interpretation_ids[0])
    assert compared == [tuple(c.id for c in group.candidates)]
    assert len(group.candidates) == 2 and group.selected_id == group.candidates[1].id
    assert turn.outcomes[0].status == 'done'
    assert (tmp_path / 'demo/main.py').is_file() and not (tmp_path / 'hello').exists()
    assert agent.interpretations.continuation_status(group.id).pending == 0


def test_new_semantically_unknown_rival_cannot_be_omitted_or_eliminated(tmp_path, monkeypatch):
    setup_source(monkeypatch)
    (tmp_path / 'current').write_text('hello')
    agent = make_agent(tmp_path)
    group_id, = agent.interpret('authored ambiguity').group_ids
    def unknown_rival(group):
        first = supplied_models(group)[0]
        return (first, CandidateHypothesis(group.candidates[1].id, (), ('unknown semantics',)))
    result = agent.resolve_interpretation(group_id, unknown_rival,
                                         max_expansions=1, max_candidates=1, max_probes=1)
    assert result.expansion.pending == 0
    assert result.investigation.decision.candidate_id is None
    assessments = result.investigation.result.assessments
    assert assessments[0].confirmed and assessments[1].viable
    assert not assessments[1].confirmed and not assessments[1].contradicted
    with pytest.raises(ValueError, match='every non-rejected'):
        agent.resolve_interpretation(group_id, lambda group: supplied_models(group)[:1])
    assert len(agent.tasks) == 0


@pytest.mark.parametrize('expansions,candidates', [(0, 1), (1, 0)])
def test_zero_expansion_or_output_budget_keeps_hidden_rival_pending(tmp_path, monkeypatch, expansions, candidates):
    setup_source(monkeypatch)
    (tmp_path / 'current').write_text('hello')
    agent = make_agent(tmp_path)
    group_id, = agent.interpret('authored ambiguity').group_ids
    result = agent.resolve_interpretation(group_id, supplied_models,
                                         max_expansions=expansions, max_candidates=candidates, max_probes=1)
    assert result.expansion is None
    assert result.investigation.decision.candidate_id is None
    assert agent.interpretations.continuation_status(group_id).pending == 1


def test_later_resolution_call_spends_new_budget_without_implicit_extra_rounds(tmp_path, monkeypatch):
    setup_source(monkeypatch, pending=2)
    (tmp_path / 'current').write_text('hello')
    agent = make_agent(tmp_path)
    group_id, = agent.interpret('authored ambiguity').group_ids
    first = agent.resolve_interpretation(group_id, supplied_models,
                                        max_expansions=1, max_candidates=1, max_probes=1)
    assert first.expansion.explored == 1 and first.expansion.pending == 1
    assert len(first.expansion.candidate_ids) == 1
    assert len(first.investigation.result.observations) == 1
    assert first.investigation.decision.candidate_id is None
    second = agent.resolve_interpretation(group_id, supplied_models,
                                         max_expansions=1, max_candidates=1, max_probes=1)
    assert second.expansion.explored == 1 and second.expansion.pending == 0
    assert len(second.investigation.result.observations) == 1
    assert second.investigation.decision.candidate_id == agent.interpretations.get(group_id).candidates[0].id
    assert len(agent.tasks) == 0
