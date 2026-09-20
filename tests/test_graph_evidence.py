"""Open-world evidence behavior on explicitly supplied graphs."""
from datetime import datetime, timezone

from tensorcode.agent.scene import SceneGraph
from tensorcode.learning.graph_evidence import assess_query
from tensorcode.learning.graph_queries import query_from_facts, match_query
from tensorcode.records import Interval, Proposition, Ref


def scene(nodes, facts):
    return SceneGraph(Ref('image:test'), tuple(nodes), tuple(facts))


def statuses(result):
    return {row.reference: row.status for row in result.roots}


def test_missing_known_root_and_unseen_referents_remain_unknown():
    a, b = Ref('node:a'), Ref('node:b')
    fact = Proposition('ready', {'item': a})
    graph = scene((a, b), (fact,))
    result = assess_query(query_from_facts((fact,), a), graph)
    assert result.complete and not result.unresolved
    assert statuses(result) == {graph.image: 'unknown', a: 'supported', b: 'unknown'}
    assert result.unseen_referents_possible
    row = next(row for row in result.roots if row.reference == a)
    assert row.supporting_matches[0].fact_indices == (0,)


def test_explicit_opposite_refutes_unary_query_without_closing_scene():
    a = Ref('node:a')
    fact = Proposition('ready', {'item': a})
    result = assess_query(query_from_facts((fact,), a), scene((a,), (
        Proposition('ready', {'item': a}, polarity=False),)))
    assert result.complete and statuses(result)[a] == 'refuted'
    row = next(row for row in result.roots if row.reference == a)
    assert row.refuting_atoms == ((0, (0,)),)
    assert result.unseen_referents_possible


def test_contradicted_existing_witness_does_not_establish_universal_refutation():
    a, b = Ref('node:a'), Ref('node:b')
    fact = Proposition('linked', {'from': a, 'to': b})
    query = query_from_facts((fact,), a)
    result = assess_query(query, scene((a, b), (
        Proposition('linked', {'from': a, 'to': b}, polarity=False),)))
    assert result.complete and statuses(result)[a] == 'unknown'
    assert not any(row.refuting_atoms for row in result.roots)
    assert result.unseen_referents_possible


def test_root_necessary_conjunct_refutes_even_when_other_witnesses_are_unknown():
    a, b = Ref('node:a'), Ref('node:b')
    required = Proposition('ready', {'item': a})
    relational = Proposition('linked', {'from': a, 'to': b})
    query = query_from_facts((required, relational), a)
    result = assess_query(query, scene((a, b), (
        Proposition('ready', {'item': a}, polarity=False),)))
    assert result.complete and statuses(result)[a] == 'refuted'
    assert statuses(result)[b] == 'unknown'


def test_required_atom_both_polarities_conflict_without_full_join_witness():
    a, b = Ref('node:a'), Ref('node:b')
    required = Proposition('ready', {'item': a})
    query = query_from_facts((required, Proposition('linked', {'from': a, 'to': b})), a)
    result = assess_query(query, scene((a, b), (required, Proposition('ready', {'item': a}, polarity=False))))
    assert result.complete and statuses(result)[a] == 'conflicted'
    assert not result.matches
    assert 'contradictory_match_evidence' in result.unresolved
    assert next(row for row in result.roots if row.reference == a).refuting_atoms


def test_conflicted_shared_variable_witness_retains_opposing_source_indices():
    a, b = Ref('node:a'), Ref('node:b')
    fact = Proposition('linked', {'from': a, 'to': b})
    result = assess_query(query_from_facts((fact,), a), scene((a, b), (
        fact, Proposition('linked', {'from': a, 'to': b}, polarity=False))))
    assert result.complete and statuses(result)[a] == 'conflicted'
    assert result.matches[0].conflicts == ((0, (1,)),)
    assert next(row for row in result.roots if row.reference == a).supporting_matches == result.matches


def test_opposite_metadata_and_typed_values_must_match_exactly():
    a = Ref('node:a')
    valid = Interval(datetime(2026, 1, 1, tzinfo=timezone.utc), None)
    positive = Proposition('ready', {'item': a, 'value': True}, valid=valid)
    alternatives = (
        Proposition('ready', {'item': a, 'value': 1}, polarity=False, valid=valid),
        Proposition('ready', {'item': a, 'value': True}, polarity=False),
        Proposition('ready', {'item': a, 'value': True}, polarity=False, valid=valid, scope=a),
        Proposition('ready', {'item': a, 'value': True}, polarity=False, valid=valid, modality='hypothesised'),
    )
    result = assess_query(query_from_facts((positive,), a), scene((a,), alternatives), max_states=10000)
    assert result.complete and statuses(result)[a] == 'unknown'
    assert not any(row.refuting_atoms for row in result.roots)


def test_shared_budget_includes_matching_and_root_assessment():
    a = Ref('node:a')
    fact = Proposition('ready', {'item': a})
    query = query_from_facts((fact,), a)
    graph = scene((a,), (fact,))
    matcher = match_query(query, graph)
    result = assess_query(query, graph)
    assert result.complete and result.explored > matcher.explored
    limited = assess_query(query, graph, max_states=matcher.explored)
    assert not limited.complete and limited.explored == matcher.explored
    assert limited.matches == matcher.matches and limited.roots == ()
    assert 'state_budget' in limited.unresolved
    assert assess_query(query, graph, max_states=result.explored) == result


def test_negative_query_is_not_proved_by_missing_positive_facts():
    a = Ref('node:a')
    negative = Proposition('ready', {'item': a}, polarity=False)
    result = assess_query(query_from_facts((negative,), a), scene((a,), ()))
    assert result.complete and statuses(result)[a] == 'unknown'


def test_whole_image_is_assessed_as_an_explicit_root():
    image = Ref('image:test')
    fact = Proposition('organized', {'scene': image})
    result = assess_query(query_from_facts((fact,), image), scene((), (fact,)))
    assert result.complete and statuses(result) == {image: 'supported'}
