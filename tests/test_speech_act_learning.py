"""Authored neutral syntax and explicit labels test learning, not a grammar seed."""
from dataclasses import replace
import pytest
from tensorcode.language import Entity, Frame, Request, Question
from tensorcode.language.deps_semantics import ProvisionalMeaning
from tensorcode.learning.speech_act import SpeechActExample, SpeechActLabel, fit_speech_acts
from tensorcode.outcomes import Unknown


def example(noun, kind='request', *, features=None, predicate='open'):
    words = (predicate, noun)
    frame = Frame(predicate, {'object': Entity('noun', noun)}, features or {})
    meaning = ProvisionalMeaning(frame, words, ('VERB', 'NOUN'), words, ((1, 0), (2, 1)),
                                 ((1, 'root'), (2, 'obj')), 1)
    label = SpeechActLabel(kind)
    return SpeechActExample(noun, 'source:' + noun, ' '.join(words), meaning, label, ('explicit supplied labels',))


def test_transfer_to_unseen_lexical_slot_preserves_neutral_frame_and_qualifiers():
    model = fit_speech_acts([example('files', features={'polarity': 'negative'}), example('folders', features={'polarity': 'negative'})],
                           [example('windows', features={'polarity': 'negative'})])
    fresh = example('documents', features={'polarity': 'negative'})
    result = model.propose(fresh.meaning)
    assert result.complete and len(result.proposals) == 1
    proposal = result.proposals[0]
    assert type(proposal.meaning) is Request and proposal.meaning.frame == fresh.meaning.frame
    assert proposal.frame.features == {'polarity': 'negative'}
    assert proposal.training_example_ids == ('files', 'folders')
    assert proposal.validation_example_ids == ('windows',)
    assert not model.propose(example('documents').meaning).proposals
    assert not model.propose(example('documents', predicate='remove', features={'polarity': 'negative'}).meaning).proposals


def test_identical_syntax_can_preserve_competing_explicit_intents():
    training = [example('files'), example('folders'), example('windows', 'statement'), example('doors', 'statement')]
    validation = [example('documents'), example('panels', 'statement')]
    model = fit_speech_acts(training, validation)
    result = model.propose(example('records').meaning)
    assert {p.label.kind for p in result.proposals} == {'request', 'statement'}
    assert all(p.conflicting_validation_example_ids for p in result.proposals)
    assert {type(p.meaning) for p in result.proposals} == {Request, Frame}


def test_unvalidated_competitor_is_visible_and_no_default_supplies_intent():
    model = fit_speech_acts([example('files'), example('folders'), example('windows', 'statement'), example('doors', 'statement')],
                           [example('documents')])
    result = model.propose(example('records').meaning)
    assert len(result.proposals) == 1 and any(reason.startswith('unvalidated_speech_act:') for reason in result.unresolved)
    assert not fit_speech_acts([example('files')], [example('folders')]).propose(example('records').meaning).proposals


def test_explicit_question_correspondence_consumes_only_taught_source_slot():
    def question(noun):
        result = example(noun)
        return replace(result, label=SpeechActLabel('question', 'object', ('roles', 'object'), (1,)))
    model = fit_speech_acts([question('files'), question('folders')], [question('windows')])
    fresh = question('documents')
    proposal = model.propose(fresh.meaning).proposals[0]
    assert type(proposal.meaning) is Question and proposal.meaning.asked == 'object'
    assert proposal.meaning.frame.roles == {}
    assert proposal.frame == fresh.meaning.frame
    assert proposal.frame.roles['object'].text == 'documents'
    assert proposal.label.query_path == ('roles', 'object') and proposal.label.token_indices == (1,)
    with pytest.raises(ValueError, match='absent'):
        bad = replace(question('files'), label=SpeechActLabel('question', 'location', ('roles', 'location'), (1,)))
        fit_speech_acts([bad], [])


def test_explicit_unresolved_label_does_not_become_statement_or_request():
    model = fit_speech_acts([example('files', 'unresolved'), example('folders', 'unresolved')], [example('windows', 'unresolved')])
    proposal = model.propose(example('documents').meaning).proposals[0]
    assert isinstance(proposal.meaning, Unknown) and proposal.label.kind == 'unresolved'


def test_source_text_identity_leakage_and_unanchored_tokens_rejected():
    first, second, held = example('files'), example('folders'), example('windows')
    with pytest.raises(ValueError, match='IDs'):
        fit_speech_acts([first, second], [replace(held, source_id=first.source_id)])
    with pytest.raises(ValueError, match='texts'):
        fit_speech_acts([first, second], [replace(held, text='OPEN   FILES')])
    with pytest.raises(ValueError, match='tokens'):
        fit_speech_acts([first, second], [replace(held, text='unrelated text')])


def test_syntax_changes_and_budget_exhaustion_preserve_uncertainty():
    model = fit_speech_acts([example('files'), example('folders')], [example('windows')])
    altered = replace(example('documents').meaning, tags=('NOUN', 'NOUN'))
    assert not model.propose(altered).proposals
    bounded = fit_speech_acts([example('files'), example('folders'), example('doors')], [example('windows')], max_pairs=1)
    assert not bounded.complete and 'pair_budget_exhausted' in bounded.unresolved
    invalid = replace(example('records').meaning, heads=((1, 2), (2, 1)))
    assert not model.propose(invalid).complete


def test_nested_taught_query_hole_preserves_other_entity_and_clause_qualifiers():
    def quantity(noun):
        words = ('count', 'how', 'many', noun)
        frame = Frame('count', {'object': Entity('description', noun, {'quantity': 'how many', 'color': 'green'})},
                      {'polarity': 'negative'})
        meaning = ProvisionalMeaning(frame, words, ('VERB', 'ADV', 'ADJ', 'NOUN'), words,
            ((1, 0), (2, 3), (3, 4), (4, 1)), ((1, 'root'), (2, 'advmod'), (3, 'amod'), (4, 'obj')), 1)
        return SpeechActExample(noun, 'source:' + noun, ' '.join(words), meaning,
            SpeechActLabel('question', 'quantity', ('roles', 'object', 'features', 'quantity'), (1, 2)))
    model = fit_speech_acts([quantity('apples'), quantity('pears')], [quantity('oranges')])
    fresh = quantity('plums')
    proposal = model.propose(fresh.meaning).proposals[0]
    assert proposal.meaning.frame.roles['object'].features == {'color': 'green'}
    assert proposal.meaning.frame.features == {'polarity': 'negative'}
    assert proposal.frame.roles['object'].features['quantity'] == 'how many'
    assert proposal.label.token_indices == (1, 2)


def test_question_consumption_rejects_wrong_span_and_qualified_target():
    first = example('files')
    wrong = replace(first, label=SpeechActLabel('question', 'object', ('roles', 'object'), (0,)))
    with pytest.raises(ValueError, match='does not correspond'):
        fit_speech_acts([wrong], [])
    qualified = replace(first.meaning, frame=Frame('open', {'object': Entity('noun', 'files', {'color': 'green'})}))
    labeled = replace(first, meaning=qualified, label=SpeechActLabel('question', 'object', ('roles', 'object'), (1,)))
    with pytest.raises(ValueError, match='qualified'):
        fit_speech_acts([labeled], [])
