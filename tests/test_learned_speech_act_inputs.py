"""Actual trained syntax feeds supervised intent, without surface-rule fallback.

Communicative labels and selection of the tested syntax candidate are supplied.
The segmentation, POS and dependency proposals come from local trained models.
This bounded corpus measures the integration, not general English intent accuracy.
"""
from pathlib import Path

import pytest

from tensorcode.agent import Agent, InterpretationDecision
from tensorcode.agent.understand import LearnedReader
from tensorcode.language.deps_semantics import ProvisionalMeaning
from tensorcode.outcomes import Unknown


@pytest.fixture(scope='module')
def actual_reader():
    root = Path.home() / '.cache/tensorcode/models'
    if not all((root / name).exists() for name in ('ud_ewt_parser.pickle', 'ud_ewt_segmenter.json')):
        pytest.skip('locally trained parsing and segmentation artifacts required')
    return LearnedReader(tag_beam_width=2, tag_max_candidates=2,
        parse_beam_width=4, parse_max_candidates=2, max_alternatives=8,
        segmentation_max_candidates=1)


def neutral_candidate(group):
    matches = []
    for candidate in group.candidates:
        for index, act in enumerate(candidate.payload.acts):
            meaning = act.meaning
            if (type(meaning) is ProvisionalMeaning and meaning.tags == ('VERB', 'NOUN')
                    and dict(meaning.labels) == {1: 'root', 2: 'obj'}
                    and dict(meaning.heads) == {1: 0, 2: 1}):
                matches.append((candidate, index))
    assert len(matches) == 1, [(c.id, c.payload.metadata) for c in group.candidates]
    return matches[0]


def retained_input(agent, text):
    message = agent.interpret(text)
    assert message.unavailable is None
    group, = [agent.interpretations.get(ident) for ident in message.group_ids]
    candidate, index = neutral_candidate(group)
    source = agent.interpretations.get_source(group.source_id)
    for anchor in candidate.payload.metadata['token_anchors']:
        start, end = anchor['char_span']
        assert source.text[start:end] == anchor['token']
    return group, candidate, index


@pytest.mark.parametrize('text', ['open files', 'running errands'])
def test_actual_subjectless_syntax_has_no_default_communicative_authority(actual_reader, text):
    def choose_neutral(group):
        candidate, _ = neutral_candidate(group)
        return InterpretationDecision(candidate.id, 'explicit fixture choice of syntax, not communicative intent',
            compared_revision=group.revision, compared_candidate_ids=tuple(c.id for c in group.candidates))
    agent = Agent([], reader=actual_reader, interpretation_selector=choose_neutral)
    group, candidate, _ = retained_input(agent, text)
    assert all(act.kind not in ('request', 'tell', 'question')
               for candidate in group.candidates for act in candidate.payload.acts)
    assert 'mood' not in candidate.payload.acts[0].meaning.frame.features
    turn = agent.turn(text)
    assert all(outcome.status in ('unknown', 'not_understood') for outcome in turn.outcomes)
    assert len(agent.tasks) == 0
    assert not agent.store.propositions() and not agent.store.claims()


def teach(agent, examples):
    from tensorcode.agent.speech_act_learning import retain_speech_act_example
    from tensorcode.learning.speech_act import SpeechActLabel
    records = []
    for text, label in examples:
        group, candidate, index = retained_input(agent, text)
        record = retain_speech_act_example(agent, group.id, candidate.id, index,
            SpeechActLabel(label), basis=('explicit communicative teaching label over actual trained syntax',))
        assert not isinstance(record, Unknown), record
        records.append(record)
    return records


def admitted(agent, training, validation):
    from tensorcode.agent.speech_act_learning import fit_speech_act_model, admit_speech_act_model
    fitted = fit_speech_act_model(agent, training, validation)
    assert not isinstance(fitted, Unknown), fitted
    handle = admit_speech_act_model(agent, fitted, reason='explicit admission after source-disjoint validation')
    assert not isinstance(handle, Unknown), handle
    agent.speech_act_model = handle
    return handle


def test_real_syntax_teaching_transfers_request_and_preserves_nonrequest(actual_reader):
    from tensorcode.agent.speech_act_learning import speech_act_dependencies
    agent = Agent([], reader=actual_reader)
    training = teach(agent, [('open files', 'request'), ('open folders', 'request'),
                            ('running errands', 'unresolved'), ('running tests', 'unresolved')])
    validation = teach(agent, [('open windows', 'request'), ('running experiments', 'unresolved')])
    handle = admitted(agent, training, validation)
    group, original, _ = retained_input(agent, 'open pictures')
    requests = [candidate for candidate in group.candidates
                if any(act.kind == 'request' for act in candidate.payload.acts)]
    assert len(requests) == 1
    learned = requests[0]
    assert learned.payload.acts[0].frame == original.payload.acts[0].meaning.frame
    assert learned.payload.acts[0].frame.roles['object'].text == 'pictures'
    dependencies = speech_act_dependencies(agent, group.id, learned.id)
    assert not isinstance(dependencies, Unknown), dependencies
    assert any(dependency.group_id == handle.group_id for dependency in dependencies)
    assert group.selected_id is None
    other, _, _ = retained_input(agent, 'running checks')
    assert all(act.kind not in ('request', 'tell', 'question')
               for candidate in other.candidates for act in candidate.payload.acts)
    assert not agent.tasks.values() and not agent.store.propositions()
    agent.interpretations.unset(handle.group_id, reason='withdraw communicative model admission')
    assert isinstance(speech_act_dependencies(agent, group.id, learned.id), Unknown)
    after, _, _ = retained_input(agent, 'open documents')
    assert all(act.kind not in ('request', 'tell', 'question')
               for candidate in after.candidates for act in candidate.payload.acts)


def test_conflicting_real_input_teaching_retains_both_communicative_interpretations(actual_reader):
    agent = Agent([], reader=actual_reader)
    training = teach(agent, [('open files', 'request'), ('open folders', 'request'),
                            ('open books', 'statement'), ('open letters', 'statement')])
    validation = teach(agent, [('open windows', 'request'), ('open doors', 'statement')])
    admitted(agent, training, validation)
    group, original, _ = retained_input(agent, 'open pictures')
    learned = [candidate for candidate in group.candidates
               if any(act.kind in ('request', 'tell') for act in candidate.payload.acts)]
    assert {candidate.payload.acts[0].kind for candidate in learned} == {'request', 'tell'}
    assert all(candidate.payload.acts[0].frame == original.payload.acts[0].meaning.frame for candidate in learned)
    assert group.selected_id is None
    turn = agent.turn('open pictures')
    assert all(outcome.status == 'unknown' for outcome in turn.outcomes)
    assert not agent.tasks.values() and not agent.store.propositions()
