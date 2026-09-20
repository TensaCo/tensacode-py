"""Connected partial witnesses are supplied evidence, never invented entities."""
from tensorcode.agent.scene import SceneGraph
from tensorcode.learning.graph_partial import match_partial_query
from tensorcode.learning.graph_queries import GraphQuery, encode_value, query_from_facts
from tensorcode.records import Interval, Proposition, Ref


A, B, C = (Ref('node:' + name) for name in 'abc')


def scene(*facts):
    return SceneGraph(Ref('image:test'), (A, B, C), tuple(facts))


def test_observed_relation_binds_existing_group_for_missing_group_property():
    relation = Proposition('member', {'item': A, 'group': B})
    missing = Proposition('radial', {'group': B})
    query = query_from_facts((relation, missing), A)
    result = match_partial_query(query, scene(relation), A)
    assert result.complete and not result.unresolved and len(result.matches) == 2
    empty, joined = result.matches
    assert empty.bindings == ((0, A),) and not empty.supporting
    assert empty.remaining == (0, 1)
    assert set(dict(joined.bindings).values()) == {A, B}
    assert dict(joined.bindings)[0] == A
    assert len(joined.supporting) == 1 and joined.supporting[0][1] == 0
    assert len(joined.remaining) == 1


def test_multistep_chain_keeps_original_variable_indices_and_exact_support():
    first = Proposition('first', {'from': A, 'to': B})
    second = Proposition('second', {'from': B, 'to': C})
    third = Proposition('property', {'item': C})
    query = query_from_facts((first, second, third), A)
    result = match_partial_query(query, scene(second, first), A)
    assert result.complete
    full = next(row for row in result.matches if len(row.supporting) == 2)
    assert set(dict(full.bindings).values()) == {A, B, C}
    assert dict(full.bindings)[0] == A
    assert {fact for _, fact in full.supporting} == {0, 1}
    assert len(full.remaining) == 1
    assert all(len(row.bindings) == len(set(ref for _, ref in row.bindings)) for row in result.matches)


def test_disconnected_fact_never_supplies_arbitrary_witness_binding():
    refs = [A, B]
    atoms = (encode_value(Proposition('root', {'item': A}), refs),
             encode_value(Proposition('elsewhere', {'item': B}), refs))
    query = GraphQuery(atoms, 2)
    result = match_partial_query(query, scene(Proposition('root', {'item': A}),
        Proposition('elsewhere', {'item': B})), A)
    assert result.complete and len(result.matches) == 2
    assert all(row.bindings == ((0, A),) for row in result.matches)
    assert all(1 in row.remaining for row in result.matches)


def test_absent_relation_does_not_enumerate_declared_nodes_as_witnesses():
    relation = Proposition('member', {'item': A, 'group': B})
    query = query_from_facts((relation, Proposition('radial', {'group': B})), A)
    result = match_partial_query(query, scene(Proposition('radial', {'group': B})), A)
    assert result.complete and len(result.matches) == 1
    assert result.matches[0].bindings == ((0, A),)
    assert not result.matches[0].supporting


def test_contradictory_join_retains_opposing_indices_without_erasing_support():
    relation = Proposition('member', {'item': A, 'group': B})
    query = query_from_facts((relation, Proposition('radial', {'group': B})), A)
    result = match_partial_query(query, scene(relation,
        Proposition('member', {'item': A, 'group': B}, polarity=False)), A)
    assert result.complete and result.unresolved == ('contradictory_match_evidence',)
    joined = next(row for row in result.matches if row.supporting)
    assert joined.conflicts == ((0, (1,)),)


def test_typed_nested_values_scope_validity_and_injectivity_remain_exact():
    relation = Proposition('member', {'item': A, 'detail': {'group': [B], 'value': True}}, scope=A)
    query = query_from_facts((relation,), A)
    near = (
        Proposition('member', {'item': A, 'detail': {'group': [B], 'value': 1}}, scope=A),
        Proposition('member', {'item': A, 'detail': {'group': [B], 'value': True}}),
        Proposition('member', {'item': A, 'detail': {'group': [A], 'value': True}}, scope=A),
        Proposition('member', relation.roles, scope=A, modality='hypothesised'),
    )
    result = match_partial_query(query, scene(*near, relation), A, max_states=10000)
    assert result.complete and len(result.matches) == 2
    assert result.matches[1].supporting == ((0, 4),)


def test_join_order_duplicates_are_deduplicated_without_dropping_partial_evidence():
    first = Proposition('first', {'item': A})
    second = Proposition('second', {'item': A})
    query = query_from_facts((first, second), A)
    result = match_partial_query(query, scene(first, second), A)
    assert result.complete and len(result.matches) == 4
    assert len({row.supporting for row in result.matches}) == 4
    assert sum(not row.remaining for row in result.matches) == 1


def test_match_and_actual_work_limits_report_incomplete_partial_search():
    relation = Proposition('member', {'item': A, 'group': B})
    query = query_from_facts((relation,), A)
    graph = scene(relation)
    limited = match_partial_query(query, graph, A, max_matches=1)
    assert not limited.complete and len(limited.matches) == 1
    assert 'match_limit' in limited.unresolved
    full = match_partial_query(query, graph, A)
    bounded = match_partial_query(query, graph, A, max_states=full.explored - 1)
    assert not bounded.complete and bounded.explored <= full.explored - 1
    assert 'state_budget' in bounded.unresolved
    assert match_partial_query(query, graph, A, max_states=full.explored) == full
