"""Bounded exact matching over supplied graphs, without a visual ontology."""
from dataclasses import dataclass
from datetime import datetime, timezone

from tensorcode.agent.scene import SceneGraph
from tensorcode.learning.graph_queries import (
    GraphQuery, encode_value, enumerate_rooted_queries, match_query, query_from_facts,
)
from tensorcode.outcomes import Unknown
from tensorcode.records import Interval, Proposition, Ref


def graph(nodes, facts, image='image:test'):
    return SceneGraph(Ref(image), tuple(Ref(n) for n in nodes), tuple(facts))


def test_conjunction_requires_join_and_retains_fact_indices():
    a, b, c = (Ref('node:' + name) for name in 'abc')
    query = query_from_facts((Proposition('rel', {'a': a, 'b': b}),
                             Proposition('attribute', {'node': b, 'value': 'blue'})), a)
    scene = graph(['node:a', 'node:b', 'node:c'], [
        Proposition('rel', {'a': a, 'b': b}),
        Proposition('attribute', {'node': c, 'value': 'blue'}),
        Proposition('attribute', {'node': b, 'value': 'blue'}),
    ])
    result = match_query(query, scene)
    assert result.complete and len(result.matches) == 1
    assert result.matches[0].bindings[0] == a
    assert set(result.matches[0].fact_indices) == {0, 2}


def test_distinct_pattern_references_cannot_collapse_to_one_node():
    a, b = Ref('node:a'), Ref('node:b')
    query = query_from_facts((Proposition('rel', {'a': a, 'b': b}),), a)
    result = match_query(query, graph(['node:a'], [Proposition('rel', {'a': a, 'b': a})]))
    assert result.complete and not result.matches


def test_scope_validity_modality_and_polarity_are_exact():
    a, scope = Ref('node:a'), Ref('node:scope')
    valid = Interval(datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 2, tzinfo=timezone.utc))
    wanted = Proposition('event', {'who': a}, False, 'hypothesised', valid, scope)
    query = query_from_facts((wanted,), a)
    facts = [Proposition('event', {'who': a}, False, 'hypothesised', valid, None),
             Proposition('event', {'who': a}, False, 'hypothesised', Interval(), scope),
             Proposition('event', {'who': a}, True, 'hypothesised', valid, scope),
             Proposition('event', {'who': a}, False, 'asserted', valid, scope), wanted]
    result = match_query(query, graph(['node:a', 'node:scope'], facts))
    assert result.complete and [m.fact_indices for m in result.matches] == [(4,)]
    unscoped = query_from_facts((Proposition('event', {'who': a}, scope=None),), a)
    assert not match_query(unscoped, graph(['node:a', 'node:scope'], [
        Proposition('event', {'who': a}, scope=scope)])).matches


def test_absence_never_establishes_a_negative_fact():
    a = Ref('node:a')
    query = query_from_facts((Proposition('near', {'node': a}, polarity=False),), a)
    result = match_query(query, graph(['node:a'], [Proposition('near', {'node': a})]))
    assert result.complete and not result.matches


def test_bool_int_float_list_and_tuple_are_not_interchangeable():
    a = Ref('node:a')
    facts = [Proposition('value', {'node': a, 'value': value}) for value in (True, 1, 1.0, [1], (1,))]
    scene = graph(['node:a'], facts)
    for index, fact in enumerate(facts):
        result = match_query(query_from_facts((fact,), a), scene)
        assert result.complete and [m.fact_indices for m in result.matches] == [(index,)]


def test_nested_references_and_metadata_participate_in_shared_bindings():
    a, b = Ref('node:a'), Ref('node:b')
    fact = Proposition('nested', {'value': {'sequence': [a, (b,)]},
                                  'report': Proposition('inside', {'who': b}, scope=a)})
    query = query_from_facts((fact,), a)
    result = match_query(query, graph(['node:a', 'node:b'], [fact]))
    assert result.complete and result.matches[0].bindings[0] == a
    assert set(result.matches[0].bindings) == {a, b}


def test_pattern_identity_is_invariant_to_reference_renaming_node_and_fact_order():
    a, b, c = Ref('node:a'), Ref('node:b'), Ref('node:c')
    x, y, z = Ref('other:z'), Ref('other:y'), Ref('other:x')
    left = [Proposition('rel', {'a': a, 'b': b}), Proposition('rel', {'a': b, 'b': c})]
    right = [Proposition('rel', {'a': y, 'b': z}), Proposition('rel', {'a': x, 'b': y})]
    assert query_from_facts(left, a) == query_from_facts(right, x)
    first = enumerate_rooted_queries(graph(['node:a', 'node:b', 'node:c'], left), a, max_atoms=2)
    second = enumerate_rooted_queries(graph(['other:x', 'other:y', 'other:z'], right), x, max_atoms=2)
    assert first.complete and second.complete
    assert set(first.patterns) == set(second.patterns)


def test_reference_mapping_keys_are_abstracted_without_name_order_authority():
    a, b, c = Ref('node:a'), Ref('node:b'), Ref('node:c')
    x, y, z = Ref('other:z'), Ref('other:y'), Ref('other:x')
    left = Proposition('map', {'root': a, 'mapping': {b: 'blue', c: 'red'}})
    right = Proposition('map', {'root': x, 'mapping': {z: 'red', y: 'blue'}})
    query = query_from_facts((left,), a)
    assert query == query_from_facts((right,), x)
    result = match_query(query, graph(['other:z', 'other:y', 'other:x'], [right]))
    assert result.complete and result.matches[0].bindings[0] == x


def test_work_budgets_bound_failed_search_as_well_as_output():
    a = Ref('node:a')
    query = query_from_facts((Proposition('absent', {'node': a}),), a)
    scene = graph(['node:a'], [Proposition('other', {'node': a, 'value': i}) for i in range(100)])
    result = match_query(query, scene, max_states=20)
    assert not result.complete and result.explored <= 20 and result.unresolved
    patterns = enumerate_rooted_queries(scene, a, max_states=20)
    assert not patterns.complete and patterns.explored <= 20 and patterns.unresolved
    assert isinstance(query_from_facts(scene.propositions[:3], a, max_states=1), Unknown)


def test_match_and_pattern_caps_do_not_claim_exhaustiveness():
    a, b = Ref('node:a'), Ref('node:b')
    facts = [Proposition('tag', {'node': a}), Proposition('tag', {'node': b})]
    scene = graph(['node:a', 'node:b'], facts)
    query = query_from_facts((facts[0],), a)
    result = match_query(query, scene, max_matches=1)
    assert len(result.matches) == 1 and not result.complete
    connected = graph(['node:a'], [Proposition('one', {'node': a}), Proposition('two', {'node': a})])
    result = enumerate_rooted_queries(connected, a, max_patterns=1)
    assert len(result.patterns) == 1 and not result.complete


def test_unsupported_typed_value_is_reported_without_repr_identity():
    @dataclass(frozen=True)
    class Opaque:
        value: int
        def __repr__(self):
            raise AssertionError('repr is not semantic identity')
    a = Ref('node:a')
    scene = graph(['node:a'], [Proposition('opaque', {'node': a, 'value': Opaque(1)})])
    query = query_from_facts((Proposition('plain', {'node': a}),), a)
    result = match_query(query, scene)
    assert not result.complete and 'unsupported graph value type: Opaque' in result.unresolved
    enumeration = enumerate_rooted_queries(scene, a)
    assert not enumeration.complete and not enumeration.patterns
    assert encode_value(True) != encode_value(1)


def test_whole_image_can_root_a_relational_organization_query():
    image, group = Ref('image:test'), Ref('node:group')
    scene = graph(['node:group'], [
        Proposition('organization', {'scene': image, 'group': group}),
        Proposition('layout', {'group': group, 'arrangement': 'radial'}),
    ])
    query = query_from_facts(scene.propositions, image)
    result = match_query(query, scene)
    assert result.complete and result.matches[0].bindings[0] == image
    enumerated = enumerate_rooted_queries(scene, image, max_atoms=2)
    assert enumerated.complete and query in enumerated.patterns


def test_malformed_query_is_unresolved_even_with_no_facts():
    scene = graph([], [])
    result = match_query(GraphQuery((('Proposition', ('var', 0)),), 1), scene)
    assert not result.complete and result.unresolved
    atom = encode_value(Proposition('event', {'node': Ref('node:a')}), [])
    result = match_query(GraphQuery((atom,), 2), scene)
    assert not result.complete and 'invalid graph query variables' in result.unresolved


def test_opposite_polarity_retains_witness_and_all_opposing_fact_indices():
    a = Ref('node:a')
    positive = Proposition('red', {'item': a})
    negative = Proposition('red', {'item': a}, polarity=False)
    scene = graph(['node:a'], [positive, negative, negative])
    result = match_query(query_from_facts((positive,), a), scene)
    assert result.complete and result.unresolved == ('contradictory_match_evidence',)
    assert result.matches[0].fact_indices == (0,)
    assert result.matches[0].conflicts == ((0, (1, 2)),)
    negative_result = match_query(query_from_facts((negative,), a), scene)
    assert negative_result.complete and len(negative_result.matches) == 2
    assert {m.conflicts for m in negative_result.matches} == {((1, (0,)),), ((2, (0,)),)}


def test_join_conflict_is_attached_to_its_supporting_atom_only():
    a, b = Ref('node:a'), Ref('node:b')
    relation = Proposition('linked', {'from': a, 'to': b})
    property_ = Proposition('ready', {'item': b})
    scene = graph(['node:a', 'node:b'], [relation, property_, Proposition('ready', {'item': b}, polarity=False)])
    result = match_query(query_from_facts((relation, property_), a), scene)
    assert result.complete and len(result.matches) == 1
    assert set(result.matches[0].fact_indices) == {0, 1}
    assert result.matches[0].bindings[0] == a
    assert result.matches[0].conflicts == ((1, (2,)),)
    assert result.unresolved == ('contradictory_match_evidence',)


def test_unrelated_contradiction_does_not_contaminate_query_evidence():
    a, b = Ref('node:a'), Ref('node:b')
    wanted = Proposition('ready', {'item': a})
    scene = graph(['node:a', 'node:b'], [wanted, Proposition('red', {'item': b}),
        Proposition('red', {'item': b}, polarity=False)])
    result = match_query(query_from_facts((wanted,), a), scene)
    assert result.complete and not result.unresolved and not result.matches[0].conflicts


def test_opposition_preserves_scope_validity_modality_nested_polarity_and_types():
    a, scope = Ref('node:a'), Ref('node:scope')
    valid = Interval(datetime(2026, 1, 1, tzinfo=timezone.utc), None)
    nested = Proposition('inside', {'item': a}, polarity=False)
    roles = {'item': a, 'nested': nested, 'value': True}
    positive = Proposition('claimed', roles, True, 'hypothesised', valid, scope)
    near_opposites = [
        Proposition('claimed', roles, False, 'hypothesised', valid, None),
        Proposition('claimed', roles, False, 'hypothesised', Interval(), scope),
        Proposition('claimed', roles, False, 'asserted', valid, scope),
        Proposition('claimed', {**roles, 'value': 1}, False, 'hypothesised', valid, scope),
        Proposition('claimed', {**roles, 'nested': Proposition('inside', {'item': a})}, False, 'hypothesised', valid, scope),
    ]
    scene = graph(['node:a', 'node:scope'], [positive, *near_opposites])
    query = query_from_facts((positive,), a)
    result = match_query(query, scene, max_states=10000)
    assert result.complete and not result.unresolved and not result.matches[0].conflicts
    exact = Proposition('claimed', roles, False, 'hypothesised', valid, scope)
    result = match_query(query, graph(['node:a', 'node:scope'], [*scene.propositions, exact]), max_states=10000)
    assert result.complete and result.matches[0].conflicts == ((0, (6,)),)


def test_contradiction_search_is_deterministic_and_budgeted():
    a = Ref('node:a')
    positive = Proposition('ready', {'item': a})
    scene = graph(['node:a'], [positive, Proposition('ready', {'item': a}, polarity=False)])
    query = query_from_facts((positive,), a)
    result = match_query(query, scene)
    assert result == match_query(query, scene)
    bounded = match_query(query, scene, max_states=result.explored - 1)
    assert not bounded.complete and bounded.explored == result.explored - 1
    assert 'state_budget' in bounded.unresolved
    assert match_query(query, scene, max_states=result.explored) == result
