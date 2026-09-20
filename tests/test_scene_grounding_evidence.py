"""Authored descriptions, scenes, and alignments test evidence integrity, not vision."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tensorcode.agent.interpretation import InterpretationWorkspace
from tensorcode.agent.scene import SceneGraph, SceneProposal
from tensorcode.agent.scene_grounding import (
    SceneGroundingModelHandle, SceneGroundingReport, RetainedGroundingExample,
    retain_grounding_example, fit_grounding_model, admit_grounding_model,
    get_grounding_model, propose_scene_groundings, grounding_dependencies,
)
from tensorcode.agent.understand import Act, SentenceAlternative
from tensorcode.language import Entity, Frame, Request
from tensorcode.outcomes import Unknown
from tensorcode.records import Ref, Proposition

PATH = ('acts', 0, 'frame', 'roles', 'object')


def agent():
    return SimpleNamespace(interpretations=InterpretationWorkspace())


def language(owner, *, surface='the relational target'):
    workspace = owner.interpretations
    source = workspace.add_source('inspect ' + surface, provider='authored test description')
    group = workspace.create_group(source.id)
    entity = Entity('description', surface, {'noun': 'target'})
    frame = Frame('inspect', {'object': entity})
    payload = SentenceAlternative(None, (Act('request', Request(frame), frame),))
    candidate = workspace.propose(group.id, payload, provenance=('authored test language',))
    return group.id, candidate.id


def scene(owner, name, *, duplicate=False, image_target=False):
    workspace = owner.interpretations
    image = Ref('image:' + name)
    target, distractor, anchor, other = (Ref(name + ':' + n) for n in ('target', 'distractor', 'anchor', 'other'))
    nodes = (target, distractor, anchor, other)
    actual_target = image if image_target else target
    facts = [Proposition('opaque-relation', {'left': actual_target, 'right': anchor}),
             Proposition('opaque-relation', {'left': distractor, 'right': other}),
             Proposition('opaque-property', {'entity': anchor, 'value': True}),
             Proposition('opaque-property', {'entity': other, 'value': False})]
    if duplicate:
        facts.append(Proposition('opaque-relation', {'left': distractor, 'right': anchor}))
    graph = SceneGraph(image, nodes, tuple(facts))
    source = workspace.add_source('supplied pixels:' + name, modality='image',
                                  metadata={'image_ref': image.id}, payload=b'authored fixture bytes')
    group = workspace.create_group(source.id, provenance=('authored test scenes',))
    candidate = workspace.propose(group.id, SceneProposal(graph, ('authored graph fixture',)))
    return group.id, candidate.id, actual_target, (distractor, anchor, other)


def teaching(owner, name, *, image_target=False):
    language_id, reading_id = language(owner)
    scene_id, graph_id, positive, negative = scene(owner, name, image_target=image_target)
    record = retain_grounding_example(owner, language_id, reading_id, PATH, scene_id, graph_id,
                                      (positive,), negative, basis=('explicit teacher alignment',))
    assert isinstance(record, RetainedGroundingExample), record
    return record


def fitted(owner, *, admit=True, image_target=False, **bounds):
    train = [teaching(owner, name, image_target=image_target) for name in ('train1', 'train2')]
    valid = [teaching(owner, 'heldout', image_target=image_target)]
    handle = fit_grounding_model(owner, train, valid, **bounds)
    assert isinstance(handle, SceneGroundingModelHandle), handle
    if admit:
        handle = admit_grounding_model(owner, handle, reason='explicit test model admission')
        assert isinstance(handle, SceneGroundingModelHandle), handle
    return handle, train, valid


def inference(owner, handle, *, select=True, duplicate=False, image_target=False):
    language_id, reading_id = language(owner)
    scene_id, graph_id, positive, negative = scene(owner, 'novel', duplicate=duplicate, image_target=image_target)
    if select:
        owner.interpretations.select(scene_id, graph_id, reason='explicit test scene interpretation')
    result = propose_scene_groundings(owner, handle, language_id, reading_id, PATH, scene_id, graph_id)
    return result, language_id, reading_id, scene_id, graph_id, positive, negative


def test_authentic_teaching_retains_exact_source_comparisons_without_selection():
    owner = agent()
    record = teaching(owner, 'teaching')
    assert record.example.basis == ('explicit teacher alignment',)
    assert record.language.group.selected_id is None and record.scene.group.selected_id is None
    source = owner.interpretations.get_source(record.evidence_source_id)
    assert source.payload == record.example
    assert source.metadata['path'] == PATH
    record.example.description.features['tampered'] = True
    train2, held = teaching(owner, 'other'), teaching(owner, 'held')
    result = fit_grounding_model(owner, [record, train2], [held])
    assert isinstance(result, Unknown) and 'modified' in result.detail


def test_fit_requires_disjoint_authentic_retained_examples():
    owner = agent()
    one, two, held = (teaching(owner, name) for name in ('one', 'two', 'held'))
    assert isinstance(fit_grounding_model(owner, [one, two], [one]), Unknown)
    forged = replace(held, example=replace(held.example, positive_refs=held.example.negative_refs[:1], negative_refs=()))
    assert isinstance(fit_grounding_model(owner, [one, two], [forged]), Unknown)
    # Returning to a prior selected state still changes the recorded comparison.
    owner.interpretations.select(one.language.group_id, one.language.candidate_id, reason='later decision')
    owner.interpretations.unset(one.language.group_id, reason='withdraw later decision')
    assert isinstance(fit_grounding_model(owner, [one, two], [held]), Unknown)


def test_fit_and_admission_are_separate_and_handles_are_authenticated():
    owner = agent()
    handle, _, _ = fitted(owner, admit=False)
    assert owner.interpretations.get(handle.group_id).selected_id is None
    assert isinstance(get_grounding_model(owner, handle), Unknown)
    assert isinstance(admit_grounding_model(owner, replace(handle, model_id='forged'), reason='test'), Unknown)
    admitted = admit_grounding_model(owner, handle, reason='explicit admission')
    assert not isinstance(get_grounding_model(owner, admitted), Unknown)
    forged = replace(admitted, dependency=replace(admitted.dependency, basis=('invented admission',)))
    assert isinstance(get_grounding_model(owner, forged), Unknown)


def test_novel_scene_publishes_every_distinct_binding_and_preserves_match_evidence():
    owner = agent()
    handle, _, _ = fitted(owner)
    result, group_id, parent_id, scene_id, graph_id, target, negative = inference(owner, handle, duplicate=True)
    assert isinstance(result, SceneGroundingReport) and result.complete, result
    assert len(result.candidate_ids) == 2
    group = owner.interpretations.get(group_id)
    assert group.selected_id is None and len(group.candidates) == 3
    children = [c for c in group.candidates if c.id in result.candidate_ids]
    refs = {c.payload.acts[0].frame.roles['object'].ref for c in children}
    assert refs == {target, negative[0]}
    assert group.candidates[0].id == parent_id
    assert group.candidates[0].payload.acts[0].frame.roles['object'].ref is None
    evidence = owner.interpretations.get_source(result.source_id)
    assert evidence.payload.matches and evidence.metadata['queries']
    assert evidence.metadata['validation_examples']
    for match in evidence.payload.matches:
        assert match.matched_proposition_indices and match.query_ids and match.validation_example_ids
    for child in children:
        dependencies = grounding_dependencies(owner, group_id, child.id)
        assert len(dependencies) == 2
        assert {d.group_id for d in dependencies} == {handle.group_id, scene_id}
    assert grounding_dependencies(owner, group_id, parent_id) == ()
    # An explicit language decision does not erase supporting model/scene commitments.
    owner.interpretations.select(group_id, children[0].id, reason='test chooses one alternative')
    assert len(grounding_dependencies(owner, group_id, children[0].id)) == 2


def test_publication_requires_selected_scene_and_exhausted_frontier():
    owner = agent()
    handle, _, _ = fitted(owner)
    result, group_id, _, scene_id, graph_id, _, _ = inference(owner, handle, select=False)
    assert isinstance(result, Unknown) and 'selected scene' in result.detail
    assert len(owner.interpretations.get(group_id).candidates) == 1
    class Pending:
        pending = 1
        def advance(self, **kwargs):
            return ()
    owner.interpretations.attach_continuation(scene_id, Pending())
    owner.interpretations.select(scene_id, graph_id, reason='selected but incomplete scene')
    parent = owner.interpretations.get(group_id).candidates[0]
    result = propose_scene_groundings(owner, handle, group_id, parent.id, PATH, scene_id, graph_id)
    assert isinstance(result, Unknown) and 'pending' in result.detail
    assert len(owner.interpretations.get(group_id).candidates) == 1


def test_scene_revision_and_child_payload_tampering_invalidate_support():
    owner = agent()
    handle, _, _ = fitted(owner)
    report, group_id, _, scene_id, _, _, _ = inference(owner, handle)
    child_id = report.candidate_ids[0]
    group = owner.interpretations._groups[group_id]
    child = next(c for c in group.candidates if c.id == child_id)
    child.payload.acts[0].frame.roles['object'].features['invented'] = True
    assert isinstance(grounding_dependencies(owner, group_id, child_id), Unknown)
    # Restoring cached content does not rescue a withdrawn scene commitment.
    owner.interpretations._groups[group_id] = replace(group, candidates=tuple(
        deepcopy(owner._scene_grounding_children[(group_id, child_id)].candidate) if c.id == child_id else c
        for c in group.candidates))
    owner.interpretations.unset(scene_id, reason='scene account withdrawn')
    assert isinstance(grounding_dependencies(owner, group_id, child_id), Unknown)


def test_historical_fit_survives_later_teaching_but_explicit_refit_invalidates_admission():
    owner = agent()
    handle, train, held = fitted(owner)
    report, group_id, _, _, _, _, _ = inference(owner, handle)
    refitted = fit_grounding_model(owner, train, held, group_id=handle.group_id)
    assert isinstance(refitted, SceneGroundingModelHandle)
    assert isinstance(get_grounding_model(owner, handle), Unknown)
    assert isinstance(grounding_dependencies(owner, group_id, report.candidate_ids[0]), Unknown)
    admitted = admit_grounding_model(owner, refitted, reason='explicit updated model admission')
    owner.interpretations.reject(train[0].language.group_id, train[0].language.candidate_id, reason='teaching interpretation changed')
    assert not isinstance(get_grounding_model(owner, admitted), Unknown)
    assert isinstance(fit_grounding_model(owner, train, held, group_id=handle.group_id), Unknown)
    assert not isinstance(get_grounding_model(owner, admitted), Unknown)


def test_incomplete_fit_is_retained_for_diagnosis_but_cannot_be_admitted():
    owner = agent()
    handle, _, _ = fitted(owner, admit=False, max_patterns=1)
    source = owner.interpretations.get_source(handle.evidence_source_id)
    assert source.payload['_complete'] is False and source.payload['_unresolved']
    assert isinstance(admit_grounding_model(owner, handle, reason='cannot override incompleteness'), Unknown)


def test_whole_image_grounding_is_supported_without_element_only_assumptions():
    owner = agent()
    handle, _, _ = fitted(owner, image_target=True)
    report, group_id, _, _, _, image, _ = inference(owner, handle, image_target=True)
    assert isinstance(report, SceneGroundingReport) and report.candidate_ids
    children = [c for c in owner.interpretations.get(group_id).candidates if c.id in report.candidate_ids]
    assert {c.payload.acts[0].frame.roles['object'].ref for c in children} == {image}


def test_mismatched_image_identity_is_rejected_at_teaching_boundary():
    owner = agent()
    language_id, reading_id = language(owner)
    scene_id, graph_id, positive, negative = scene(owner, 'original')
    group = owner.interpretations.get(scene_id)
    source = owner.interpretations._sources[group.source_id]
    owner.interpretations._sources[source.id] = replace(source, metadata={'image_ref': 'image:another'})
    record = retain_grounding_example(owner, language_id, reading_id, PATH, scene_id, graph_id,
                                      (positive,), negative, basis=('explicit teaching',))
    assert isinstance(record, Unknown) and 'retained image source' in record.detail


def test_incomplete_novel_scene_search_retains_diagnostics_without_language_bindings():
    owner = agent()
    handle, _, _ = fitted(owner, max_matches=256)
    language_id, parent_id = language(owner)
    image = Ref('image:large-unseen-scene')
    nodes = tuple(Ref('large:' + str(index)) for index in range(60))
    facts = tuple(Proposition('opaque-relation', {'left': node, 'right': nodes[(index + 1) % len(nodes)]})
                  for index, node in enumerate(nodes))
    facts += tuple(Proposition('opaque-property', {'entity': node, 'value': True}) for node in nodes)
    source = owner.interpretations.add_source('large scene fixture', modality='image', metadata={'image_ref': image.id})
    group = owner.interpretations.create_group(source.id)
    candidate = owner.interpretations.propose(group.id, SceneProposal(SceneGraph(image, nodes, facts)))
    owner.interpretations.select(group.id, candidate.id, reason='explicit test scene')
    report = propose_scene_groundings(owner, handle, language_id, parent_id, PATH, group.id, candidate.id)
    assert isinstance(report, SceneGroundingReport) and not report.complete
    assert report.candidate_ids == () and any('budget' in reason for reason in report.unresolved)
    assert len(owner.interpretations.get(language_id).candidates) == 1
    evidence = owner.interpretations.get_source(report.source_id)
    assert evidence.payload.complete is False and evidence.metadata['queries']


def test_teaching_revision_during_fit_prevents_model_publication(monkeypatch):
    import tensorcode.agent.scene_grounding as module
    owner = agent()
    train = [teaching(owner, 'one'), teaching(owner, 'two')]
    held = [teaching(owner, 'held')]
    original = module.fit_scene_grounding
    def changing_fit(*args, **kwargs):
        model = original(*args, **kwargs)
        owner.interpretations.unset(train[0].language.group_id, reason='teacher changed during computation')
        return model
    monkeypatch.setattr(module, 'fit_scene_grounding', changing_fit)
    assert isinstance(fit_grounding_model(owner, train, held), Unknown)
    assert not getattr(owner, '_scene_grounding_models', {})


def test_failed_refit_retains_prior_authentic_version_and_rejects_new_version(monkeypatch):
    owner = agent()
    handle, train, held = fitted(owner)
    original = owner.interpretations.propose
    changed = False
    def changing_publication(group_id, payload, **kwargs):
        nonlocal changed
        candidate = original(group_id, payload, **kwargs)
        if group_id == handle.group_id and not changed:
            changed = True
            owner.interpretations.unset(train[0].scene.group_id, reason='teacher changed during model publication')
        return candidate
    monkeypatch.setattr(owner.interpretations, 'propose', changing_publication)
    result = fit_grounding_model(owner, train, held, group_id=handle.group_id)
    assert isinstance(result, Unknown)
    group = owner.interpretations.get(handle.group_id)
    assert len(group.candidates) == 2 and group.candidates[-1].rejected
    rejected = group.candidates[-1]
    rejected_handle = SceneGroundingModelHandle(group.id, rejected.id, rejected.payload.snapshot['_id'], rejected.payload.evidence_source_id)
    assert isinstance(admit_grounding_model(owner, rejected_handle, reason='reject failed fit'), Unknown)
    readmitted = admit_grounding_model(owner, handle, reason='readmit intact historical fit')
    assert isinstance(readmitted, SceneGroundingModelHandle), readmitted
    assert not isinstance(get_grounding_model(owner, readmitted), Unknown)


def test_model_admission_cannot_capture_a_different_selection_epoch(monkeypatch):
    owner = agent()
    handle, _, _ = fitted(owner, admit=False)
    original = owner.interpretations.select
    def select_twice(group_id, candidate_id, **kwargs):
        result = original(group_id, candidate_id, **kwargs)
        original(group_id, candidate_id, reason='reentrant later decision')
        return result
    monkeypatch.setattr(owner.interpretations, 'select', select_twice)
    result = admit_grounding_model(owner, handle, reason='initial admission')
    assert isinstance(result, Unknown) and 'changed during selection' in result.detail
    assert not getattr(owner, '_scene_grounding_admissions', {})


def test_model_and_search_evidence_tampering_invalidate_learned_children():
    owner = agent()
    handle, _, _ = fitted(owner)
    report, group_id, _, _, _, _, _ = inference(owner, handle)
    source = owner.interpretations._sources[report.source_id]
    source.metadata['queries'] = ()
    assert isinstance(grounding_dependencies(owner, group_id, report.candidate_ids[0]), Unknown)
    owner.interpretations._groups[handle.group_id].selected.payload.snapshot['_complete'] = False
    assert isinstance(get_grounding_model(owner, handle), Unknown)


def test_scene_changed_during_child_publication_rejects_published_child(monkeypatch):
    owner = agent()
    handle, _, _ = fitted(owner)
    language_id, parent_id = language(owner)
    scene_id, graph_id, _, _ = scene(owner, 'new-source')
    owner.interpretations.select(scene_id, graph_id, reason='initial scene commitment')
    original = owner.interpretations.propose
    def propose(group_id, payload, **kwargs):
        candidate = original(group_id, payload, **kwargs)
        if group_id == language_id:
            owner.interpretations.unset(scene_id, reason='scene revised during publication')
        return candidate
    monkeypatch.setattr(owner.interpretations, 'propose', propose)
    result = propose_scene_groundings(owner, handle, language_id, parent_id, PATH, scene_id, graph_id)
    assert isinstance(result, Unknown)
    group = owner.interpretations.get(language_id)
    assert len(group.candidates) == 2 and group.candidates[-1].rejected
    assert isinstance(grounding_dependencies(owner, language_id, group.candidates[-1].id), Unknown)


def test_unresolved_query_rival_blocks_binding_even_with_validated_matches():
    owner = agent()
    def ambiguous_teaching(name, *, counterexample=False):
        language_id, reading_id = language(owner)
        image, target, distractor = Ref('image:' + name), Ref(name + ':target'), Ref(name + ':other')
        graph = SceneGraph(image, (target, distractor), (
            Proposition('color', {'entity': target, 'value': 'red'}),
            Proposition('size', {'entity': target, 'value': 'small' if counterexample else 'large'}),
            Proposition('color', {'entity': distractor, 'value': 'blue'}),
            Proposition('size', {'entity': distractor, 'value': 'large' if counterexample else 'small'}),
        ))
        source = owner.interpretations.add_source('pixels', modality='image', metadata={'image_ref': image.id})
        group = owner.interpretations.create_group(source.id)
        candidate = owner.interpretations.propose(group.id, SceneProposal(graph))
        record = retain_grounding_example(owner, language_id, reading_id, PATH, group.id, candidate.id,
                                          (target,), (distractor,), basis=('explicit alignment',))
        assert isinstance(record, RetainedGroundingExample), record
        return record
    train = [ambiguous_teaching('first'), ambiguous_teaching('second')]
    held = [ambiguous_teaching('held', counterexample=True)]
    fitted_handle = fit_grounding_model(owner, train, held, max_atoms=1)
    handle = admit_grounding_model(owner, fitted_handle, reason='admit competing learned queries for inspection')
    assert isinstance(handle, SceneGroundingModelHandle), handle
    fresh = ambiguous_teaching('fresh')
    owner.interpretations.select(fresh.scene.group_id, fresh.scene.candidate_id, reason='explicit scene')
    result = propose_scene_groundings(owner, handle, fresh.language.group_id, fresh.language.candidate_id,
                                     PATH, fresh.scene.group_id, fresh.scene.candidate_id)
    assert isinstance(result, SceneGroundingReport) and result.complete
    assert result.candidate_ids == () and result.unresolved
    retained = owner.interpretations.get_source(result.source_id).payload
    assert retained.matches, 'a validated match must not hide an unresolved competing query'
    assert len(owner.interpretations.get(fresh.language.group_id).candidates) == 1


def test_authenticated_grounding_descendants_keep_model_and_scene_dependencies():
    from tensorcode.agent.grounding import MentionBinding, propose_grounding
    owner = agent()
    handle, _, _ = fitted(owner)
    report, group_id, _, scene_id, _, target, _ = inference(owner, handle)
    learned_id = report.candidate_ids[0]
    child = propose_grounding(owner.interpretations, group_id, learned_id, (
        MentionBinding(PATH, target, (report.source_id,), 'compatible explicit binding'),))
    expected = grounding_dependencies(owner, group_id, learned_id)
    assert grounding_dependencies(owner, group_id, child.id) == expected
    assert {dep.group_id for dep in expected} == {handle.group_id, scene_id}
    owner.interpretations.unset(handle.group_id, reason='withdraw model admission')
    assert isinstance(grounding_dependencies(owner, group_id, child.id), Unknown)


def test_tampered_or_cyclic_grounding_derivation_is_not_trusted():
    from tensorcode.agent.grounding import MentionBinding, propose_grounding
    owner = agent()
    handle, _, _ = fitted(owner)
    report, group_id, _, _, _, target, _ = inference(owner, handle)
    child = propose_grounding(owner.interpretations, group_id, report.candidate_ids[0], (
        MentionBinding(PATH, target, (report.source_id,), 'compatible explicit binding'),))
    lineage = owner.interpretations._grounding_derivations[child.id]
    owner.interpretations._grounding_derivations[child.id] = (group_id, child.id, lineage[2])
    assert isinstance(grounding_dependencies(owner, group_id, child.id), Unknown)
    owner.interpretations._grounding_derivations[child.id] = (group_id, 'missing-parent', lineage[2])
    assert isinstance(grounding_dependencies(owner, group_id, child.id), Unknown)
    owner.interpretations._grounding_derivations[child.id] = (group_id, lineage[1], replace(lineage[2], provenance=('forged',)))
    assert isinstance(grounding_dependencies(owner, group_id, child.id), Unknown)


def test_ancestor_validation_cannot_change_descendant_comparison_unnoticed(monkeypatch):
    from tensorcode.agent.grounding import MentionBinding, propose_grounding
    owner = agent()
    handle, _, _ = fitted(owner)
    report, group_id, _, _, _, target, _ = inference(owner, handle)
    child = propose_grounding(owner.interpretations, group_id, report.candidate_ids[0], (
        MentionBinding(PATH, target, (report.source_id,), 'compatible explicit binding'),))
    original = owner.interpretations.get_source
    changed = False
    def get_source(source_id):
        nonlocal changed
        source = original(source_id)
        if source_id == handle.evidence_source_id and not changed:
            changed = True
            owner.interpretations.unset(group_id, reason='language comparison changed during ancestor validation')
        return source
    monkeypatch.setattr(owner.interpretations, 'get_source', get_source)
    result = grounding_dependencies(owner, group_id, child.id)
    assert isinstance(result, Unknown) and 'changed' in result.detail
