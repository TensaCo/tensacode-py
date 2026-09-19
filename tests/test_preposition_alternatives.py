"""Authored dependency trees isolate revisable preposition-role interpretation."""

import json
import math
from dataclasses import asdict

import pytest

from tensorcode.language.deps_semantics import Reader, SemanticProjectionIssue, preposition_roles


def tree(words, tags, heads, labels):
    return words, tags, words, dict(enumerate(heads, 1)), dict(enumerate(labels, 1))


SINGLE = tree(['sit', 'on', 'desk'], ['VERB', 'ADP', 'NOUN'], [0, 3, 1], ['root', 'case', 'obl'])
DOUBLE = tree(['sit', 'on', 'desk', 'and', 'wait', 'on', 'Tuesday'],
              ['VERB', 'ADP', 'NOUN', 'CCONJ', 'VERB', 'ADP', 'PROPN'],
              [0, 3, 1, 5, 1, 7, 5], ['root', 'case', 'obl', 'cc', 'conj', 'case', 'obl'])
PRIORS = {'on': [('location', -0.2), ('time', -1.7)]}


def test_each_occurrence_has_independent_alternatives_and_evidence():
    reader = Reader(PRIORS, preposition_provenance='authored-test:two-role-inventory')
    result = reader.read_candidates(*DOUBLE)
    assert result.complete and not result.truncated and result.pending == 0
    assert len(result.candidates) == 4
    assignments = {tuple((c.dependent_token, c.role) for c in row.choices) for row in result.candidates}
    assert assignments == {
        ((3, 'location'), (7, 'location')), ((3, 'location'), (7, 'time')),
        ((3, 'time'), (7, 'location')), ((3, 'time'), (7, 'time')),
    }
    for candidate in result.candidates:
        assert not candidate.unresolved and len(candidate.meanings) == 2
        for choice, meaning in zip(candidate.choices, candidate.meanings):
            assert choice.role in meaning.frame.roles
            assert choice.provenance == 'authored-test:two-role-inventory'
            assert choice.log_prior == dict(PRIORS['on'])[choice.role]
    json.dumps([asdict(c) for c in result.candidates[0].choices])
    assert not reader._role_bindings


def test_legacy_read_does_not_choose_most_frequent_role():
    with pytest.raises(ValueError, match='read_candidates'):
        Reader(PRIORS).read(*SINGLE)


def test_unknown_preposition_retains_anchor_without_location_assertion():
    reader = Reader({})
    result = reader.read_candidates(*SINGLE)
    [candidate] = result.candidates
    assert candidate.meanings == ()
    [issue] = candidate.unresolved
    assert (issue.dependent_token, issue.preposition) == (3, 'on')
    assert result.complete  # Enumeration complete is not semantic understanding.
    with pytest.raises(ValueError, match='read_candidates'):
        reader.read(*SINGLE)


def test_no_case_oblique_is_unresolved_instead_of_default_location():
    bare = tree(['wait', 'Tuesday'], ['VERB', 'PROPN'], [0, 1], ['root', 'obl'])
    [candidate] = Reader({}).read_candidates(*bare).candidates
    assert not candidate.meanings
    assert candidate.unresolved[0].dependent_token == 2
    assert candidate.unresolved[0].preposition == ''


def test_authored_dependency_subtype_mapping_remains_explicit():
    temporal = tree(['wait', 'Tuesday'], ['VERB', 'PROPN'], [0, 1], ['root', 'obl:tmod'])
    [meaning] = Reader({}).read(*temporal)
    assert 'time' in meaning.frame.roles


def test_colliding_role_occurrences_are_deferred_with_both_choices():
    repeated = tree(['sit', 'on', 'desk', 'on', 'chair'], ['VERB', 'ADP', 'NOUN', 'ADP', 'NOUN'],
                    [0, 3, 1, 5, 1], ['root', 'case', 'obl', 'case', 'obl'])
    result = Reader(PRIORS).read_candidates(*repeated)
    assert len(result.candidates) == 4
    collided = [c for c in result.candidates if c.unresolved]
    assert len(collided) == 2
    for candidate in collided:
        assert not candidate.meanings and len(candidate.choices) == 2
        [issue] = candidate.unresolved
        assert isinstance(issue, SemanticProjectionIssue)
        assert issue.dependent_tokens == (3, 5)
    assert len([c for c in result.candidates if c.meanings]) == 2


@pytest.mark.parametrize('bounds', [{'max_candidates': 1}, {'max_expansions': 1}, {'max_expansions': 3}])
def test_budget_exhaustion_is_reported_without_implied_exhaustiveness(bounds):
    result = Reader(PRIORS).read_candidates(*DOUBLE, **bounds)
    assert result.truncated and not result.complete and result.pending > 0
    assert result.reason
    assert len(result.candidates) <= bounds.get('max_candidates', 32)
    assert result.explored <= bounds.get('max_expansions', 256)


def test_reader_does_not_mutate_or_reuse_bindings_between_calls():
    priors = {'on': [('location', -0.2), ('time', -1.7)]}
    reader = Reader(priors)
    priors['on'].clear()
    assert len(reader.read_candidates(*SINGLE).candidates) == 2
    assert reader.read_candidates(*SINGLE) == reader.read_candidates(*SINGLE)
    with pytest.raises(TypeError):
        reader.prepositions['on'] = ()


def test_role_frequencies_use_training_data_only(tmp_path):
    def write(split, labels):
        path = tmp_path / split
        path.mkdir()
        sentences = [{'swes': {str(i): {'ss': f'p.{role}', 'lexlemma': 'on'}
                                for i, role in enumerate(labels)}}]
        (path / f'streusle.ud_{split}.json').write_text(json.dumps(sentences))
    write('train', ['Locus', 'Locus', 'Time'])
    write('dev', ['Recipient'] * 30)
    write('test', ['Instrument'] * 40)
    got = dict(preposition_roles(tmp_path)['on'])
    assert got == {'location': round(math.log(2 / 3), 3), 'time': round(math.log(1 / 3), 3)}


def test_authored_singleton_role_is_recorded_not_hidden():
    reader = Reader({'on': [('time', 0.0)]})
    [candidate] = reader.read_candidates(*SINGLE).candidates
    assert candidate.choices[0].provenance == 'authored-preposition-priors'
    assert candidate.choices[0].role == 'time'
    assert reader.read(*SINGLE) == list(candidate.meanings)


@pytest.mark.parametrize('relation', ['nmod', 'obl'])
def test_no_case_noun_modifier_does_not_become_possession(relation):
    source = tree(['inspect', 'folder', 'Tuesday'], ['VERB', 'NOUN', 'PROPN'],
                  [0, 1, 2], ['root', 'obj', relation])
    [candidate] = Reader({}).read_candidates(*source).candidates
    assert candidate.meanings == ()
    [issue] = candidate.unresolved
    assert (issue.dependent_token, issue.preposition) == (3, '')
    assert source[0][issue.dependent_token - 1] == 'Tuesday'


def test_no_case_modifier_in_naming_clause_remains_unresolved():
    from types import SimpleNamespace

    source = tree(['make', 'folder', 'called', 'notes', 'Tuesday'],
                  ['VERB', 'NOUN', 'VERB', 'NOUN', 'PROPN'],
                  [0, 1, 2, 3, 4], ['root', 'obj', 'acl', 'obj', 'nmod'])
    reader = Reader({})
    # An authored class fixture isolates naming projection from lexical inference.
    reader._verbs = {'called': (SimpleNamespace(id='dub-fixture'),)}
    [candidate] = reader.read_candidates(*source).candidates
    assert candidate.meanings == ()
    [issue] = candidate.unresolved
    assert (issue.dependent_token, issue.preposition) == (5, '')
    assert source[0][issue.dependent_token - 1] == 'Tuesday'


def test_fold_naming_cannot_supply_possessor_for_an_unmarked_relation(monkeypatch):
    from tensorcode.language import Entity

    reader = Reader({})
    source = tree(['called', 'notes', 'Tuesday'], ['VERB', 'NOUN', 'PROPN'],
                  [0, 1, 2], ['root', 'obj', 'nmod'])
    words, tags, lemmas, heads, labels = source
    # Isolate the naming projection's own relation handling from recursive entity reading.
    monkeypatch.setattr(reader, 'entity', lambda *args, **kwargs: Entity('name', 'notes'))
    with pytest.raises(ValueError, match='read_candidates'):
        reader._fold_naming(1, {}, words, tags, lemmas, heads, labels, reader.children(heads))
