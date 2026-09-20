"""Targeted hypothetical observations over explicitly authored query fixtures."""
from datetime import date, datetime, timezone

from tensorcode.agent.scene import SceneGraph
from tensorcode.learning.graph_queries import encode_value, query_from_facts
from tensorcode.learning.grounding_probes import propose_grounding_probes
from tensorcode.learning.scene_grounding import LearnedQuery, SceneGroundingModel, _description
from tensorcode.records import Interval, Proposition, Ref


ROOT = Ref('node:target')
OTHER = Ref('node:other')
DESCRIPTION = {'mention': 'chosen object'}


def model(*atom_sets):
    queries = tuple(LearnedQuery('query:' + str(i), _description(DESCRIPTION),
        query_from_facts(atoms, ROOT), ('training:1', 'training:2'), ('validation:1',), ())
        for i, atoms in enumerate(atom_sets))
    return SceneGroundingModel((), (), queries, True, (), 4096)


def scene(*facts):
    return SceneGraph(Ref('image:test'), (ROOT, OTHER), tuple(facts))


def status(rows, root=ROOT):
    return {ident: next(row.status for row in evidence.roots if row.reference == root)
            for ident, evidence in rows}


def test_unary_missing_question_predicts_both_answers_without_observing():
    wanted = Proposition('ready', {'item': ROOT})
    graph = scene()
    learned = model((wanted,))
    plan = propose_grounding_probes(learned, DESCRIPTION, graph, ROOT)
    assert plan.complete and not plan.unresolved and len(plan.probes) == 1
    assert plan.model_id == learned.id and plan.scene_id == graph.image and plan.root == ROOT
    assert status(plan.query_evidence) == {'query:0': 'unknown'}
    probe = plan.probes[0]
    assert probe.proposition == wanted and probe.query_ids == ('query:0',)
    assert probe.atom_indices == (('query:0', 0),)
    assert status(probe.positive_evidence) == {'query:0': 'supported'}
    assert status(probe.negative_evidence) == {'query:0': 'refuted'}
    assert probe.positive_evidence[0][1].unseen_referents_possible
    assert graph.propositions == ()


def test_shared_questions_retain_all_query_origins_and_conjunction_predictions():
    red = Proposition('red', {'item': ROOT})
    ready = Proposition('ready', {'item': ROOT})
    plan = propose_grounding_probes(model((red,), (red, ready)), DESCRIPTION, scene(), ROOT)
    assert plan.complete and len(plan.probes) == 2
    shared = next(probe for probe in plan.probes if probe.proposition.predicate == 'red')
    assert shared.query_ids == ('query:0', 'query:1')
    assert {ident for ident, _ in shared.atom_indices} == {'query:0', 'query:1'}
    assert status(shared.positive_evidence) == {'query:0': 'supported', 'query:1': 'unknown'}
    assert status(shared.negative_evidence) == {'query:0': 'refuted', 'query:1': 'refuted'}


def test_existing_either_polarity_produces_no_probe():
    wanted = Proposition('ready', {'item': ROOT})
    learned = model((wanted,))
    for present in (wanted, Proposition('ready', {'item': ROOT}, polarity=False)):
        plan = propose_grounding_probes(learned, DESCRIPTION, scene(present), ROOT)
        assert plan.complete and not plan.probes


def test_missing_relational_witness_is_unresolved_without_invented_reference():
    wanted = Proposition('linked', {'from': ROOT, 'to': OTHER})
    plan = propose_grounding_probes(model((wanted,)), DESCRIPTION, scene(), ROOT)
    assert plan.complete and not plan.probes
    assert plan.unresolved == ('unresolved_relational_witness:query:0:0',)
    assert status(plan.query_evidence) == {'query:0': 'unknown'}


def test_codec_preserves_full_metadata_nested_types_and_only_flips_top_polarity():
    nested = Proposition('inner', {'item': ROOT}, polarity=False)
    valid = Interval(datetime(2026, 1, 1, tzinfo=timezone.utc), None)
    wanted = Proposition('custom', {'item': ROOT, 'nested': nested,
        'payload': {ROOT: [True, 1, 1.0, ('text', date(2026, 1, 2))]}},
        False, 'hypothesised', valid, ROOT)
    plan = propose_grounding_probes(model((wanted,)), DESCRIPTION, scene(), ROOT)
    assert plan.complete and len(plan.probes) == 1
    probe = plan.probes[0]
    assert encode_value(probe.proposition) == encode_value(wanted)
    assert status(probe.positive_evidence) == {'query:0': 'supported'}
    assert status(probe.negative_evidence) == {'query:0': 'refuted'}
    assert probe.proposition.roles['nested'].polarity is False


def test_incomplete_limits_expose_no_usable_probes():
    first = Proposition('first', {'item': ROOT})
    second = Proposition('second', {'item': ROOT})
    learned = model((first, second))
    cap = propose_grounding_probes(learned, DESCRIPTION, scene(), ROOT, max_probes=1)
    assert not cap.complete and not cap.probes and 'probe_limit' in cap.unresolved
    full = propose_grounding_probes(learned, DESCRIPTION, scene(), ROOT)
    assert full.complete and len(full.probes) == 2
    limited = propose_grounding_probes(learned, DESCRIPTION, scene(), ROOT, max_states=full.explored - 1)
    assert not limited.complete and not limited.probes and 'state_budget' in limited.unresolved
    assert limited.explored <= full.explored - 1


def test_unvalidated_query_provenance_is_retained_without_admission_claim():
    wanted = Proposition('ready', {'item': ROOT})
    query = LearnedQuery('query:unvalidated', _description(DESCRIPTION),
        query_from_facts((wanted,), ROOT), ('training:1',), (), ('validation:conflict',))
    learned = SceneGroundingModel((), (), (query,), True, (), 4096)
    plan = propose_grounding_probes(learned, DESCRIPTION, scene(), ROOT)
    assert plan.complete and len(plan.probes) == 1
    assert plan.query_info == (query,)
    assert not plan.query_info[0].validation_example_ids
    assert plan.query_info[0].conflicting_validation_example_ids == ('validation:conflict',)


def test_answers_assess_relational_rivals_that_did_not_originate_question():
    ready = Proposition('ready', {'item': ROOT})
    link = Proposition('linked', {'from': ROOT, 'to': OTHER})
    neighbor_ready = Proposition('ready', {'item': OTHER})
    learned = model((ready,), (link, neighbor_ready))
    graph = scene(link, Proposition('linked', {'from': OTHER, 'to': ROOT}))
    plan = propose_grounding_probes(learned, DESCRIPTION, graph, ROOT)
    assert plan.complete and len(plan.probes) == 1
    probe = plan.probes[0]
    assert probe.query_ids == ('query:0',)
    assert status(probe.positive_evidence, OTHER)['query:1'] == 'supported'
    assert status(probe.negative_evidence, OTHER)['query:1'] == 'unknown'
    assert any(reason.startswith('unresolved_relational_witness:query:1:') for reason in plan.unresolved)


def test_existing_conjunct_is_not_requested_when_another_conjunct_is_missing():
    ready = Proposition('ready', {'item': ROOT})
    red = Proposition('red', {'item': ROOT})
    plan = propose_grounding_probes(model((ready, red)), DESCRIPTION, scene(ready), ROOT)
    assert plan.complete and len(plan.probes) == 1
    assert plan.probes[0].proposition == red
