from pathlib import Path
from tensorcode.agent import RefinementLibrary
"""Interpretation choices affect dispatch without turning alternatives into beliefs."""
import json
from dataclasses import replace

import pytest

from tensorcode.agent import Agent, FileSystemPlugin, InterpretationDecision
from tensorcode.agent.operations import Transcript
from tensorcode.agent.understand import SentenceAlternative
from interpretation_fixtures import project_sentence
from tensorcode import ops
from tensorcode.outcomes import Unknown


def project_readings(agent, monkeypatch):
    first = project_sentence('hello')
    second = project_sentence('demo')
    alternatives = tuple(SentenceAlternative(s.reading, s.acts, s.skipped, s.guessed,
                                            'test-proposal') for s in (first, second))
    sentence = replace(first, alternatives=alternatives)
    monkeypatch.setattr(ops, 'parse', lambda *a, **kw: Transcript((sentence,), 'test-reader'))


def test_interpret_retains_original_source_without_acting(tmp_path):
    agent = Agent([FileSystemPlugin(tmp_path, refinements=RefinementLibrary.load(
        Path(__file__).parent / "fixtures" / "project_refinements.json"))])
    original = '"make a python project called hello"'
    result = agent.interpret(original)
    assert agent.interpretations.get_source(result.source_id).text == original
    group = agent.interpretations.get(result.group_ids[0])
    assert group.selected_id is None and not group.history
    assert group.candidates
    assert all(a.kind == 'mention' for c in group.candidates for a in c.payload.acts)
    assert not list(tmp_path.iterdir())
    assert not agent.turns


def test_selecting_alternative_changes_actual_executed_project(tmp_path, monkeypatch):
    agent = Agent([FileSystemPlugin(tmp_path, refinements=RefinementLibrary.load(
        Path(__file__).parent / "fixtures" / "project_refinements.json"))], interpretation_selector=lambda group:
                  InterpretationDecision(group.candidates[1].id, 'external evidence identifies second'))
    project_readings(agent, monkeypatch)
    turn = agent.turn('create the intended project')
    assert [o.status for o in turn.outcomes] == ['done']
    assert (tmp_path / 'demo' / 'main.py').is_file()
    assert not (tmp_path / 'hello').exists()
    outcome = turn.outcomes[0]
    group = agent.interpretations.get(outcome.interpretation_id)
    assert outcome.candidate_id == group.selected_id == group.candidates[1].id
    assert group.history[-1].reason == 'external evidence identifies second'
    assert len(group.candidates) == 2
    json.dumps(turn.events)


def test_defer_prevents_request_execution_and_retains_candidates(tmp_path, monkeypatch):
    agent = Agent([FileSystemPlugin(tmp_path, refinements=RefinementLibrary.load(
        Path(__file__).parent / "fixtures" / "project_refinements.json"))], interpretation_selector=lambda group:
                  InterpretationDecision(None, 'need evidence about intended project'))
    project_readings(agent, monkeypatch)
    turn = agent.turn('create the intended project')
    assert not list(tmp_path.iterdir())
    assert [o.status for o in turn.outcomes] == ['unknown']
    assert 'need evidence' in turn.reply
    group = agent.interpretations.get(turn.interpretation_ids[0])
    assert group.selected_id is None and len(group.candidates) == 2


def test_explicit_selection_and_revision_do_not_replay_effects(tmp_path, monkeypatch):
    agent = Agent([FileSystemPlugin(tmp_path, refinements=RefinementLibrary.load(
        Path(__file__).parent / "fixtures" / "project_refinements.json"))], interpretation_selector=lambda group:
                  InterpretationDecision(group.candidates[0].id, "explicit test interpretation"))
    project_readings(agent, monkeypatch)
    turn = agent.turn('create the intended project')
    group = agent.interpretations.get(turn.interpretation_ids[0])
    assert group.selected_id == group.candidates[0].id
    assert group.history[0].reason == 'explicit test interpretation'
    agent.interpretations.select(group.id, group.candidates[1].id, reason='later correction')
    assert (tmp_path / 'hello' / 'main.py').is_file()
    assert not (tmp_path / 'demo').exists()
    assert turn.outcomes[0].candidate_id == group.candidates[0].id


def test_invalid_selection_cannot_dispatch(tmp_path, monkeypatch):
    agent = Agent([FileSystemPlugin(tmp_path, refinements=RefinementLibrary.load(
        Path(__file__).parent / "fixtures" / "project_refinements.json"))], interpretation_selector=lambda group:
                  InterpretationDecision('unrelated-reading', 'invalid selection'))
    project_readings(agent, monkeypatch)
    with pytest.raises(KeyError):
        agent.turn('create the intended project')
    assert not list(tmp_path.iterdir())


def test_reader_abstention_retains_source(monkeypatch):
    agent = Agent()
    monkeypatch.setattr(ops, 'parse', lambda *a, **kw: Unknown('no_reader', 'unavailable'))
    result = agent.interpret('uninterpretable original')
    assert result.unavailable.reason == 'no_reader'
    assert result.group_ids == ()
    assert agent.interpretations.get_source(result.source_id).text == 'uninterpretable original'


def test_deferred_statement_does_not_enter_belief_store():
    agent = Agent(interpretation_selector=lambda group:
                  InterpretationDecision(None, 'reference remains ambiguous'))
    turn = agent.turn('the cat is on the mat')
    assert turn.outcomes[0].status == 'unknown'
    assert agent.store.propositions() == []
    assert agent.interpretations.get(turn.interpretation_ids[0]).candidates


def test_default_never_executes_first_candidate(tmp_path, monkeypatch):
    agent = Agent([FileSystemPlugin(tmp_path, refinements=RefinementLibrary.load(
        Path(__file__).parent / "fixtures" / "project_refinements.json"))])
    project_readings(agent, monkeypatch)
    turn = agent.turn('create the intended project')
    assert not list(tmp_path.iterdir())
    assert turn.outcomes[0].status == 'unknown'
    group = agent.interpretations.get(turn.interpretation_ids[0])
    assert group.selected_id is None
    assert len(group.candidates) == 2
    assert 'no interpretation policy' in group.history[-1].reason
