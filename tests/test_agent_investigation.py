"""Fresh observations distinguish supplied meanings before real execution."""
from dataclasses import replace
from pathlib import Path

import pytest

from tensorcode import ops
from tensorcode.agent import Agent, CandidateHypothesis, Condition, FileSystemPlugin, RefinementLibrary
from tensorcode.agent.operations import Transcript
from tensorcode.agent.understand import SentenceAlternative
from interpretation_fixtures import project_sentence


def proposals(monkeypatch):
    sentences = tuple(project_sentence(name) for name in ('hello', 'demo'))
    alternatives = tuple(SentenceAlternative(s.reading, s.acts, provenance='supplied interpretation')
                         for s in sentences)
    sentence = replace(sentences[0], alternatives=alternatives)
    monkeypatch.setattr(ops, 'parse', lambda *a, **kw: Transcript((sentence,), 'supplied'))


def models(group):
    return tuple(CandidateHypothesis(candidate.id,
                    (Condition('content', {'path': 'current', 'text': name}),),
                    ('supplied semantics: named project is the workspace recorded in current',))
                 for candidate, name in zip(group.candidates, ('hello', 'demo')))


def make_agent(root, **kwargs):
    library = RefinementLibrary.load(Path(__file__).parent / 'fixtures/project_refinements.json')
    return Agent([FileSystemPlugin(root, refinements=library)], **kwargs)


def test_observed_marker_changes_executed_interpretation(tmp_path, monkeypatch):
    proposals(monkeypatch)
    (tmp_path / 'current').write_text('demo')
    agent = make_agent(tmp_path, interpretation_hypotheses=models)
    turn = agent.turn('create the project for the current workspace')
    assert turn.outcomes[0].status == 'done'
    assert (tmp_path / 'demo/main.py').is_file()
    assert not (tmp_path / 'hello').exists()
    group = agent.interpretations.get(turn.interpretation_ids[0])
    assert group.selected_id == group.candidates[1].id
    evidence_id, = group.history[-1].evidence_ids
    source = agent.interpretations.get_source(evidence_id)
    assert source.modality == 'observation'
    assert source.payload['result'].observations
    assert source.payload['result'].assessments[0].contradicted
    event = next(e for e in turn.events if e['type'] == 'interpretation_selection')
    assert event['evidence'] == [evidence_id]


def test_unexplained_marker_defers_instead_of_eliminating_to_default(tmp_path, monkeypatch):
    proposals(monkeypatch)
    (tmp_path / 'current').write_text('unmodeled workspace')
    agent = make_agent(tmp_path, interpretation_hypotheses=models)
    turn = agent.turn('create the project for the current workspace')
    assert turn.outcomes[0].status == 'unknown'
    assert not (tmp_path / 'hello').exists() and not (tmp_path / 'demo').exists()
    group = agent.interpretations.get(turn.interpretation_ids[0])
    assert group.selected_id is None
    assert group.history[-1].reason == 'all_hypotheses_contradicted'


def test_reinvestigation_withdraws_selection_and_retains_execution_history(tmp_path, monkeypatch):
    proposals(monkeypatch)
    marker = tmp_path / 'current'
    marker.write_text('hello')
    agent = make_agent(tmp_path, interpretation_hypotheses=models)
    turn = agent.turn('create the project')
    assert turn.outcomes[0].status == 'done'
    group_id = turn.interpretation_ids[0]
    old = agent.interpretations.get(group_id)
    marker.write_text('neither')
    investigation = agent.investigate_interpretation(group_id, models(old))
    updated = agent.interpretations.get(group_id)
    assert updated.selected_id is None
    assert len(updated.history) == 2
    assert updated.history[0] == old.history[0]
    assert updated.history[-1].evidence_ids == (investigation.evidence_source_id,)
    assert (tmp_path / 'hello/main.py').exists()
    assert agent.tasks.get(turn.outcomes[0].task_id).status == 'done'
    assert len(agent.turns) == 1


def test_omitting_unmodeled_rival_cannot_create_a_unique_winner(tmp_path, monkeypatch):
    proposals(monkeypatch)
    agent = make_agent(tmp_path)
    message = agent.interpret('ambiguous')
    group = agent.interpretations.get(message.group_ids[0])
    with pytest.raises(ValueError, match='every non-rejected'):
        agent.investigate_interpretation(group.id, models(group)[:1])
    assert agent.interpretations.get(group.id).revision == 0


def test_budget_exhaustion_defers(tmp_path, monkeypatch):
    proposals(monkeypatch)
    (tmp_path / 'current').write_text('demo')
    agent = make_agent(tmp_path, interpretation_hypotheses=models, interpretation_probe_budget=0)
    turn = agent.turn('create project')
    assert turn.outcomes[0].status == 'unknown'
    assert not (tmp_path / 'demo').exists()


def test_revision_cannot_cite_missing_evidence(tmp_path, monkeypatch):
    proposals(monkeypatch)
    agent = make_agent(tmp_path)
    message = agent.interpret('ambiguous')
    group = agent.interpretations.get(message.group_ids[0])
    with pytest.raises(KeyError):
        agent.interpretations.select(group.id, group.candidates[0].id,
                                     reason='unsupported source', evidence_ids=('source:missing',))
    assert agent.interpretations.get(group.id).revision == 0
