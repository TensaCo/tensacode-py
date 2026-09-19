"""Metric correctness with authored predictions, not learned understanding claims."""
from types import SimpleNamespace

import pytest

from eval.parsing.span_evaluation import Arc, Group, gold_arcs, parse_record, reader_groups, score, summarize


def record(text, rows):
    return parse_record('# sent_id = fixture\n# text = ' + text + '\n' + '\n'.join(rows))


def row(i, form, head, label, upos='NOUN'):
    return f'{i}\t{form}\t_\t{upos}\t_\t_\t{head}\t{label}\t_\t_'


def test_original_contractions_and_repeated_words_have_distinct_exact_spans():
    gold = record("I'm here, and I'm happy.", [
        row('1-2', "I'm", '_', '_', '_'), row(1, 'I', 3, 'nsubj'), row(2, "'m", 3, 'cop', 'AUX'),
        row(3, 'here', 0, 'root'), row(4, ',', 3, 'punct', 'PUNCT'), row(5, 'and', 8, 'cc'),
        row('6-7', "I'm", '_', '_', '_'), row(6, 'I', 8, 'nsubj'), row(7, "'m", 8, 'cop', 'AUX'),
        row(8, 'happy', 3, 'conj'), row(9, '.', 3, 'punct', 'PUNCT')])
    assert gold.error is None
    assert gold.spans[1] == (0, 1) and gold.spans[6] == (14, 15)
    assert gold.text[slice(*gold.spans[7])] == "'m"
    got = score(gold, (Group((0, len(gold.text)), (gold_arcs(gold),)),))
    aggregate = summarize([got])
    assert aggregate['oracle']['las'] == aggregate['oracle']['word_alignment_recall'] == 1
    assert aggregate['oracle']['conditional_words'] == 7


def test_merged_quotation_loses_only_affected_arcs_and_words():
    gold = record('She calls "Ada" today.', [row(1, 'She', 2, 'nsubj'), row(2, 'calls', 0, 'root'),
        row(3, '"', 4, 'punct', 'PUNCT'), row(4, 'Ada', 2, 'obj'), row(5, '"', 4, 'punct', 'PUNCT'),
        row(6, 'today', 2, 'advmod'), row(7, '.', 2, 'punct', 'PUNCT')])
    arcs = gold_arcs(gold)
    candidate = (arcs[0], arcs[1], Arc((10, 15), arcs[1].span, 'obj'), arcs[5], arcs[6])
    result = summarize([score(gold, (Group((0, len(gold.text)), (candidate,)),))])
    assert result['oracle']['las_correct'] == 3
    assert result['oracle']['las'] == .75
    assert result['oracle']['conditional_words'] == 3
    assert result['oracle']['conditional_las'] == 1
    assert result['oracle']['word_alignment_recall'] == 4 / 7
    assert not result['oracle']['exact_tree_recall']


def test_oracle_cannot_mix_incompatible_word_choices_within_a_candidate():
    gold = record('Birds fly.', [row(1, 'Birds', 2, 'nsubj'), row(2, 'fly', 0, 'root'), row(3, '.', 2, 'punct', 'PUNCT')])
    a, b, punctuation = gold_arcs(gold)
    left = (a, Arc(b.span, None, 'wrong'), punctuation)
    right = (Arc(a.span, b.span, 'wrong'), b, punctuation)
    got = score(gold, (Group((0, len(gold.text)), (left, right)),))
    assert got['oracle']['las_correct'] == 1  # per-token cherry picking would give 2.
    assert not got['oracle']['exact_tree']


def test_sentence_split_cannot_make_wrong_root_correct():
    gold = record('Birds fly.', [row(1, 'Birds', 2, 'nsubj'), row(2, 'fly', 0, 'root'), row(3, '.', 2, 'punct', 'PUNCT')])
    a, b, punctuation = gold_arcs(gold)
    groups = (Group((0, 5), ((Arc(a.span, None, 'root'),),)), Group((6, 10), ((b, punctuation),)))
    got = score(gold, groups)
    assert got['oracle']['uas_correct'] == 1
    assert not got['oracle']['exact_tree']


def test_inconsistent_multiword_form_and_missing_source_are_explicit_errors():
    bad = record('cannot', [row('1-2', 'cannot', '_', '_', '_'), row(1, 'can', 0, 'root'), row(2, 'never', 1, 'advmod')])
    assert 'non-concatenative' in bad.error
    result = summarize([score(bad, ())])
    assert result['dataset_alignment_errors'] == 1 and result['oracle']['las'] == 0
    absent = parse_record('# sent_id = missing\n' + row(1, 'word', 0, 'root'))
    assert absent.error == 'missing original # text'


def test_empty_nodes_excluded_and_zero_denominators_are_not_perfect_scores():
    gold = record('.', [row(1, '.', 0, 'root', 'PUNCT'), row('1.1', 'implicit', '_', '_')])
    assert gold.empty_nodes == 1 and len(gold.words) == 1 and gold.error is None
    result = summarize([score(gold, (Group((0, 1), (gold_arcs(gold),)),))])
    assert result['oracle']['uas'] is None and result['oracle']['conditional_las'] is None
    assert result['oracle']['word_alignment_recall'] == 1


def reading(anchors=None, heads=None):
    metadata = {'syntax_complete': True, 'sentence_span': (0, 10),
                'tokens': ('Birds', 'fly', '.'),
                'token_anchors': anchors or ({'index': 1, 'token': 'Birds', 'char_span': (0, 5)},
                    {'index': 2, 'token': 'fly', 'char_span': (6, 9)}, {'index': 3, 'token': '.', 'char_span': (9, 10)}),
                'heads': heads or {1: 2, 2: 0, 3: 2}, 'labels': {1: 'nsubj', 2: 'root', 3: 'punct'}}
    return SimpleNamespace(text='Birds fly.', tokens=('Birds', 'fly', '.'), alternatives=(SimpleNamespace(metadata=metadata),))


@pytest.mark.parametrize('bad_anchor', [
    {'index': 1, 'token': 'Birds', 'char_span': (1, 6)},
    {'index': 0, 'token': 'Birds', 'char_span': (0, 5)},
    {'index': True, 'token': 'Birds', 'char_span': (0, 5)},
    {'index': 2, 'token': 'Birds', 'char_span': (0, 5)},
])
def test_invalid_anchor_source_or_index_never_receives_credit(bad_anchor):
    r = reading()
    metadata = r.alternatives[0].metadata
    metadata['token_anchors'] = (bad_anchor, *metadata['token_anchors'][1:])
    groups, errors = reader_groups('Birds fly.', [r])
    assert not groups and errors


def test_duplicate_overlapping_anchors_and_groups_are_rejected():
    r = reading()
    anchors = r.alternatives[0].metadata['token_anchors']
    r.alternatives[0].metadata['token_anchors'] = (anchors[0], anchors[0], anchors[2])
    assert reader_groups('Birds fly.', [r])[1]
    groups, errors = reader_groups('Birds fly.', [reading(), reading()])
    assert not groups and 'overlapping' in errors[0]


def test_duplicate_semantic_proposals_do_not_duplicate_syntax_credit():
    r = reading()
    r.alternatives += r.alternatives
    groups, errors = reader_groups('Birds fly.', [r])
    assert not errors and len(groups[0].candidates) == 1


def test_each_alternative_has_its_own_segmentation_not_the_display_tokens():
    r = reading()
    r.tokens = ('Birds fly.',)  # Display tokens cannot constrain another candidate.
    groups, errors = reader_groups('Birds fly.', [r])
    assert not errors and len(groups[0].candidates[0]) == 3
    del r.alternatives[0].metadata['tokens']
    assert reader_groups('Birds fly.', [r])[1]  # Missing provenance is explicit.


def test_multiword_row_cannot_reassign_a_previously_anchored_word():
    gold = record('a ab', [row(1, 'a', 0, 'root'), row('1-2', 'ab', '_', '_', '_'), row(2, 'b', 1, 'dep')])
    assert gold.error and not gold.spans


@pytest.mark.parametrize('mutation', ['bool_head', 'bool_span', 'missing_token'])
def test_malformed_candidate_metadata_is_explicit_not_silently_scoreable(mutation):
    r = reading()
    metadata = r.alternatives[0].metadata
    if mutation == 'bool_head':
        metadata['heads'][1] = True
    elif mutation == 'bool_span':
        metadata['sentence_span'] = (False, 10)
    else:
        metadata['token_anchors'] = metadata['token_anchors'][:2]
    groups, errors = reader_groups('Birds fly.', [r])
    assert not groups and errors
