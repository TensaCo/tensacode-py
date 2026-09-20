"""Distinct acts share source evidence, never independent validation evidence."""
from dataclasses import replace
import pytest
from tensorcode.language import Entity, Frame, Request
from tensorcode.language.deps_semantics import ProvisionalMeaning
from tensorcode.learning.speech_act import SpeechActExample, SpeechActLabel, fit_speech_acts


def utterance(noun):
    words = ('open', noun, 'keep', 'notes')
    frames = (Frame('open', {'object': Entity('noun', noun)}),
              Frame('keep', {'object': Entity('noun', 'notes')}))
    return tuple(SpeechActExample(noun + ':' + str(i), 'source:' + noun, ' '.join(words),
        ProvisionalMeaning(frame, words, ('VERB', 'NOUN', 'VERB', 'NOUN'), words,
            ((1, 0), (2, 1), (3, 1), (4, 3)),
            ((1, 'root'), (2, 'obj'), (3, 'conj'), (4, 'obj')), 1, i),
        SpeechActLabel('request')) for i, frame in enumerate(frames))


def test_each_act_learns_from_independent_utterances_without_discarding_siblings():
    model = fit_speech_acts((*utterance('files'), *utterance('folders')), utterance('windows'))
    fresh = utterance('documents')
    for item in fresh:
        batch = model.propose(item.meaning)
        assert batch.complete and len(batch.proposals) == 1
        assert type(batch.proposals[0].meaning) is Request
        assert batch.proposals[0].frame == item.meaning.frame
        assert len(batch.proposals[0].training_example_ids) == 2
        assert len(batch.proposals[0].validation_example_ids) == 1


def test_one_utterance_cannot_supply_both_training_and_validation_acts():
    first = utterance('files')
    with pytest.raises(ValueError, match='IDs'):
        fit_speech_acts((first[0], *utterance('folders')), (first[1],))


def test_duplicate_act_cannot_inflate_teaching_even_with_a_new_example_id():
    first = utterance('files')
    with pytest.raises(ValueError, match='IDs'):
        fit_speech_acts((*first, replace(first[0], id='duplicate')), utterance('windows'))


def test_same_source_cannot_change_text_or_dependency_reading_between_acts():
    first = utterance('files')
    changed = replace(first[1], meaning=replace(first[1].meaning, tags=('VERB', 'PROPN', 'VERB', 'NOUN')))
    with pytest.raises(ValueError, match='source'):
        fit_speech_acts((first[0], changed, *utterance('folders')), utterance('windows'))


def test_duplicate_text_with_new_source_id_is_not_independent_evidence():
    first = utterance('files')
    fake = tuple(replace(e, id='fake:' + e.id, source_id='source:fake') for e in first)
    with pytest.raises(ValueError, match='texts'):
        fit_speech_acts((*first, *fake), utterance('windows'))


def phrase_example(color):
    words = ('open', 'the', color, 'folder')
    frame = Frame('open', {'object': Entity('noun', 'the ' + color + ' folder',
        {'definite': True, 'modifier': color})})
    meaning = ProvisionalMeaning(frame, words, ('VERB', 'DET', 'ADJ', 'NOUN'), words,
        ((1, 0), (2, 4), (3, 4), (4, 1)),
        ((1, 'root'), (2, 'det'), (3, 'amod'), (4, 'obj')), 1)
    return SpeechActExample(color, 'source:' + color, ' '.join(words), meaning, SpeechActLabel('request'))


def test_multiword_entity_text_shares_source_lexical_bindings_and_retains_qualifiers():
    model = fit_speech_acts((phrase_example('blue'), phrase_example('red')), (phrase_example('green'),))
    fresh = phrase_example('yellow')
    batch = model.propose(fresh.meaning)
    assert batch.complete and len(batch.proposals) == 1
    assert batch.proposals[0].frame.roles['object'].text == 'the yellow folder'
    assert batch.proposals[0].frame.roles['object'].features == {'definite': True, 'modifier': 'yellow'}
    inconsistent = replace(fresh.meaning, frame=Frame('open', {
        'object': replace(fresh.meaning.frame.roles['object'], text='the purple folder')}))
    assert not model.propose(inconsistent).proposals
