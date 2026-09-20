"""Supported bindings coexist with explicit, nonexecutable open-world alternatives."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tensorcode.agent.grounding import MentionBinding, propose_grounding
from tensorcode.agent.interpretation import InterpretationWorkspace
from tensorcode.agent.scene import SceneGraph, SceneProposal
from tensorcode.agent.scene_grounding import (
    UnresolvedGrounding, SceneGroundingReport, retain_grounding_example,
    fit_grounding_model, admit_grounding_model, propose_scene_groundings,
    grounding_dependencies,
)
from tensorcode.agent.understand import Act, SentenceAlternative
from tensorcode.language import Entity, Frame, Request
from tensorcode.outcomes import Unknown
from tensorcode.records import Ref, Proposition

PATH = ('acts', 0, 'frame', 'roles', 'object')


def language(owner):
    source = owner.interpretations.add_source('inspect the marked thing', provider='authored test reading')
    group = owner.interpretations.create_group(source.id)
    frame = Frame('inspect', {'object': Entity('description', 'the marked thing')})
    candidate = owner.interpretations.propose(group.id, SentenceAlternative(None, (Act('request', Request(frame), frame),)))
    return group.id, candidate.id


def scene(owner, name, *, refute_b=False, refute_image=False, conflict=False, extra_marker=False):
    image, a, b = (Ref(name + ':' + label) for label in ('image', 'a', 'b'))
    facts = [Proposition('opaque-mark', {'entity': a})]
    if extra_marker:
        facts.append(Proposition('opaque-extra', {'entity': a}))
        if refute_b:
            facts.append(Proposition('opaque-extra', {'entity': b}, polarity=False))
    if refute_b:
        facts.append(Proposition('opaque-mark', {'entity': b}, polarity=False))
    if refute_image:
        facts.append(Proposition('opaque-mark', {'entity': image}, polarity=False))
    if conflict:
        facts.append(Proposition('opaque-mark', {'entity': a}, polarity=False))
    graph = SceneGraph(image, (a, b), tuple(facts))
    source = owner.interpretations.add_source('supplied image', modality='image',
        metadata={'image_ref': image.id}, payload=b'authored scene fixture, not learned pixels')
    group = owner.interpretations.create_group(source.id)
    candidate = owner.interpretations.propose(group.id, SceneProposal(graph))
    return group.id, candidate.id, graph


def setup(*, extra_marker=False):
    owner = SimpleNamespace(interpretations=InterpretationWorkspace())
    examples = []
    for name in ('train-one', 'train-two', 'heldout'):
        lg, lc = language(owner)
        sg, sc, graph = scene(owner, name, refute_b=True, extra_marker=extra_marker)
        example = retain_grounding_example(owner, lg, lc, PATH, sg, sc,
            (graph.nodes[0],), (graph.nodes[1],), basis=('authored positive and negative alignment',))
        assert not isinstance(example, Unknown), example
        examples.append(example)
    model = fit_grounding_model(owner, examples[:2], examples[2:], max_atoms=1)
    model = admit_grounding_model(owner, model, reason='explicit fixture model admission')
    assert not isinstance(model, Unknown), model
    return owner, model


def infer(owner, model, **options):
    lg, lc = language(owner)
    sg, sc, graph = scene(owner, 'novel', **options)
    owner.interpretations.select(sg, sc, reason='explicit scene interpretation')
    result = propose_scene_groundings(owner, model, lg, lc, PATH, sg, sc)
    return result, lg, lc, sg, graph


def unresolved(owner, report, group_id):
    group = owner.interpretations.get(group_id)
    return [candidate for candidate in group.candidates if candidate.id in report.unresolved_candidate_ids]


def test_supported_a_does_not_erase_unknown_b_or_unseen_referents():
    owner, model = setup()
    report, lg, parent, sg, graph = infer(owner, model)
    assert isinstance(report, SceneGroundingReport) and report.complete, report
    group = owner.interpretations.get(lg)
    assert group.selected_id is None
    bound = [candidate for candidate in group.candidates if candidate.id in report.candidate_ids]
    assert len(bound) == 1 and bound[0].payload.acts[0].frame.roles['object'].ref == graph.nodes[0]
    unknown = unresolved(owner, report, lg)
    assert all(type(candidate.payload) is UnresolvedGrounding for candidate in unknown)
    assert {candidate.payload.reference for candidate in unknown} == {graph.nodes[1], graph.image, None}
    assert all(candidate.payload.evidence_source_id == report.source_id for candidate in unknown)
    assert all(candidate.payload.query_ids and candidate.payload.reason for candidate in unknown)
    assert parent not in report.candidate_ids and parent not in report.unresolved_candidate_ids
    assert set(report.candidate_ids).isdisjoint(report.unresolved_candidate_ids)
    source = owner.interpretations.get_source(report.source_id)
    assert source.payload.query_evidence
    assert any(root.reference == graph.nodes[1] and root.status == 'unknown'
               for _, evidence in source.payload.query_evidence for root in evidence.roots)
    assert {dep.group_id for dep in grounding_dependencies(owner, lg, bound[0].id)} == {model.group_id, sg}


def test_explicit_opposites_remove_known_unknowns_but_never_close_unseen_world():
    owner, model = setup()
    report, lg, _, _, graph = infer(owner, model, refute_b=True, refute_image=True)
    assert isinstance(report, SceneGroundingReport) and report.complete
    unknown = unresolved(owner, report, lg)
    assert len(unknown) == 1 and unknown[0].payload.reference is None
    source = owner.interpretations.get_source(report.source_id)
    assert all(root.status in ('supported', 'refuted')
               for _, evidence in source.payload.query_evidence for root in evidence.roots)
    assert all(evidence.unseen_referents_possible for _, evidence in source.payload.query_evidence)


def test_unresolved_alternative_has_authentic_evidence_but_no_executable_dependency():
    owner, model = setup()
    report, lg, _, _, graph = infer(owner, model)
    candidate = next(c for c in unresolved(owner, report, lg) if c.payload.reference == graph.nodes[1])
    result = grounding_dependencies(owner, lg, candidate.id)
    assert isinstance(result, Unknown) and result.reason == 'grounding_unresolved'
    owner.interpretations.select(lg, candidate.id, reason='explicitly keep uncertainty unresolved')
    result = grounding_dependencies(owner, lg, candidate.id)
    assert isinstance(result, Unknown) and result.reason == 'grounding_unresolved'
    with pytest.raises(TypeError, match='SentenceAlternative'):
        propose_grounding(owner.interpretations, lg, candidate.id, (
            MentionBinding(PATH, graph.nodes[1], (report.source_id,), 'cannot turn uncertainty into an identity'),))


def test_tampered_unresolved_candidate_is_detected_before_reporting_uncertainty():
    owner, model = setup()
    report, lg, _, _, graph = infer(owner, model)
    candidate = unresolved(owner, report, lg)[0]
    group = owner.interpretations._groups[lg]
    forged = replace(candidate, payload=replace(candidate.payload, reference=graph.nodes[0]))
    owner.interpretations._groups[lg] = replace(group, candidates=tuple(
        forged if child.id == candidate.id else child for child in group.candidates))
    result = grounding_dependencies(owner, lg, candidate.id)
    assert isinstance(result, Unknown) and result.reason == 'scene_grounding_dependency_changed'


@pytest.mark.parametrize('change', ['scene', 'report', 'language'])
def test_support_change_during_uncertainty_publication_rejects_all_children(monkeypatch, change):
    owner, model = setup()
    lg, lc = language(owner)
    sg, sc, _ = scene(owner, 'novel')
    owner.interpretations.select(sg, sc, reason='explicit scene')
    original = owner.interpretations.propose
    changed = False
    def propose(group_id, payload, **kwargs):
        nonlocal changed
        candidate = original(group_id, payload, **kwargs)
        if type(payload) is UnresolvedGrounding and not changed:
            changed = True
            if change == 'scene':
                owner.interpretations.unset(sg, reason='scene revised during uncertainty publication')
            elif change == 'language':
                owner.interpretations.select(lg, lc, reason='reentrant reading selection during publication')
            else:
                owner.interpretations._sources[payload.evidence_source_id].metadata['tampered'] = True
        return candidate
    monkeypatch.setattr(owner.interpretations, 'propose', propose)
    result = propose_scene_groundings(owner, model, lg, lc, PATH, sg, sc)
    assert isinstance(result, Unknown), result
    group = owner.interpretations.get(lg)
    assert len(group.candidates) > 2
    assert all(candidate.rejected for candidate in group.candidates if candidate.id != lc)


def test_contradictory_scene_keeps_existing_diagnostic_only_behavior():
    owner, model = setup()
    report, lg, _, _, _ = infer(owner, model, conflict=True)
    assert isinstance(report, SceneGroundingReport) and report.unresolved
    assert report.candidate_ids == () and report.unresolved_candidate_ids == ()
    assert len(owner.interpretations.get(lg).candidates) == 1


def test_unknown_root_aggregates_all_query_ids_without_duplicate_unknown_bindings():
    owner, model = setup(extra_marker=True)
    report, lg, _, _, graph = infer(owner, model, extra_marker=True)
    assert isinstance(report, SceneGroundingReport) and report.complete
    unknown_b = [candidate for candidate in unresolved(owner, report, lg)
                 if candidate.payload.reference == graph.nodes[1]]
    assert len(unknown_b) == 1 and len(unknown_b[0].payload.query_ids) == 2
    unknown_unseen = [candidate for candidate in unresolved(owner, report, lg)
                      if candidate.payload.reference is None]
    assert len(unknown_unseen) == 1 and len(unknown_unseen[0].payload.query_ids) == 2


def test_manually_proposed_uncertainty_cannot_authorize_a_direct_request():
    owner = SimpleNamespace(interpretations=InterpretationWorkspace())
    lg, _ = language(owner)
    source = owner.interpretations.add_source('explicit unknown reference evidence')
    candidate = owner.interpretations.propose(lg, UnresolvedGrounding(
        Ref('unresolved:manual'), ('query:manual',), 'identity not established', source.id))
    owner.interpretations.select(lg, candidate.id, reason='explicitly retain uncertainty')
    result = grounding_dependencies(owner, lg, candidate.id)
    assert isinstance(result, Unknown) and result.reason == 'grounding_unresolved'


def test_unregistered_candidate_inspection_checks_reentrant_comparison_changes(monkeypatch):
    owner = SimpleNamespace(interpretations=InterpretationWorkspace())
    lg, reading = language(owner)
    original = owner.interpretations.get
    def get(group_id):
        group = original(group_id)
        owner.interpretations.unset(group_id, reason='changed while inspecting unregistered reading')
        return group
    monkeypatch.setattr(owner.interpretations, 'get', get)
    result = grounding_dependencies(owner, lg, reading)
    assert isinstance(result, Unknown) and 'comparison changed' in result.detail
