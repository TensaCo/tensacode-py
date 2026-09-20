"""Explicit teacher answers revise retained models; no natural-language teacher is inferred."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tensorcode.agent.interpretation import InterpretationWorkspace
from tensorcode.agent.grounding_investigation import (
    GroundingFeedback, GroundingInvestigationProposal, prepare_grounding_investigation,
    record_grounding_feedback, refit_grounding_from_feedback,
)
from tensorcode.agent.scene_grounding import SceneGroundingModelHandle, admit_grounding_model, get_grounding_model
from tensorcode.outcomes import Unknown
from tensorcode.records import Ref
from test_agent_grounding_investigation import case, model_with_correlated_teaching, PATH


def prepared():
    owner = SimpleNamespace(interpretations=InterpretationWorkspace())
    model = model_with_correlated_teaching(owner)
    offered = case(owner, 'crossed', color=0, shape=1)
    proposal = prepare_grounding_investigation(owner, model, offered[0], offered[1], PATH, ((offered[2], offered[3]),))
    assert isinstance(proposal, GroundingInvestigationProposal), proposal
    return owner, model, offered, proposal


def answered():
    owner, model, offered, proposal = prepared()
    feedback = record_grounding_feedback(owner, proposal, offered[4].image,
        (offered[4].nodes[0],), (offered[4].nodes[1],), basis=('explicit teacher alignment',))
    assert isinstance(feedback, GroundingFeedback), feedback
    return owner, model, offered, proposal, feedback


def test_predictions_precede_feedback_and_are_preserved_with_query_provenance():
    owner, model, offered, proposal = prepared()
    prior = owner.interpretations.get_source(proposal.evidence_source_id)
    assert prior.metadata['feedback_observed'] is False
    assert not getattr(owner, '_grounding_investigation_feedback', {})
    predictions = proposal.investigation.predictions[0].predictions
    assert len({frozenset(p.references) for p in predictions}) == 3
    assert all(p.training_example_ids and p.validation_example_ids for p in predictions)
    feedback = record_grounding_feedback(owner, proposal, offered[4].image,
        (offered[4].nodes[0],), (), basis=('positive only; other nodes are unlabeled',))
    assert isinstance(feedback, GroundingFeedback), feedback
    assert feedback.teaching_record.example.negative_refs == ()
    current = owner.interpretations.get_source(proposal.evidence_source_id)
    assert current == prior
    response = owner.interpretations.get_source(feedback.evidence_source_id)
    assert response.metadata['prediction_source_id'] == proposal.evidence_source_id
    assert response.metadata['predictions'] == proposal.investigation.predictions[0]
    assert {ident for ident, pattern in response.metadata['query_patterns']} == {p.query_id for p in predictions}
    assert all(pattern.atoms for ident, pattern in response.metadata['query_patterns'])
    assert set(feedback.confirmed_query_ids) | set(feedback.contradicted_query_ids) == {p.query_id for p in predictions}
    assert owner.interpretations.get(model.group_id).selected_id == model.candidate_id


def test_forged_returned_proposal_does_not_poison_canonical_record():
    owner, _, offered, proposal = prepared()
    authentic = deepcopy(proposal)
    proposal.investigation.description.features['invented'] = True
    assert isinstance(record_grounding_feedback(owner, proposal, offered[4].image,
        (offered[4].nodes[0],), (), basis=('explicit',)), Unknown)
    accepted = record_grounding_feedback(owner, authentic, offered[4].image,
        (offered[4].nodes[0],), (), basis=('explicit',))
    assert isinstance(accepted, GroundingFeedback), accepted


@pytest.mark.parametrize('change', ['model', 'language', 'scene', 'pixels', 'predictions'])
def test_changed_model_or_original_evidence_cannot_receive_feedback(change):
    owner, model, offered, proposal = prepared()
    if change == 'model':
        owner.interpretations.unset(model.group_id, reason='withdraw model')
    elif change == 'language':
        owner.interpretations.select(offered[0], offered[1], reason='new interpretation epoch')
    elif change == 'scene':
        owner.interpretations.unset(offered[2], reason='withdraw scene')
    elif change == 'pixels':
        source_id = owner.interpretations.get(offered[2]).source_id
        source = owner.interpretations._sources[source_id]
        owner.interpretations._sources[source_id] = replace(source, payload=b'different pixels')
    else:
        source = owner.interpretations._sources[proposal.evidence_source_id]
        source.metadata['feedback_observed'] = True
    result = record_grounding_feedback(owner, proposal, offered[4].image,
        (offered[4].nodes[0],), (), basis=('explicit',))
    assert isinstance(result, Unknown)
    assert not getattr(owner, '_grounding_investigation_feedback', {})


def test_training_and_heldout_scenes_are_excluded_before_ranking():
    owner, model, offered, _ = prepared()
    fit_source = owner.interpretations.get_source(model.evidence_source_id)
    held = owner._scene_grounding_examples[fit_source.metadata['heldout_evidence_ids'][0]][0]
    excluded = (held.scene.group_id, held.scene.candidate_id)
    bad = prepare_grounding_investigation(owner, model, offered[0], offered[1], PATH, (excluded,))
    assert isinstance(bad, Unknown) and 'disjoint' in bad.detail
    valid = prepare_grounding_investigation(owner, model, offered[0], offered[1], PATH,
                                           (excluded, (offered[2], offered[3])))
    assert isinstance(valid, GroundingInvestigationProposal)
    assert tuple(s.image for s in valid.investigation.scenes) == (offered[4].image,)
    source = owner.interpretations.get_source(valid.evidence_source_id)
    assert source.metadata['excluded_scenes'][0][:2] == excluded
    denied = record_grounding_feedback(owner, valid, held.example.scene.image,
        held.example.positive_refs, held.example.negative_refs, basis=('cannot recycle heldout',))
    assert isinstance(denied, Unknown)


def test_teacher_can_choose_nonbest_offered_scene_without_hidden_ranking_authority():
    owner, model, crossing, _ = prepared()
    redundant = case(owner, 'redundant', color=0, shape=0)
    proposal = prepare_grounding_investigation(owner, model, crossing[0], crossing[1], PATH,
        ((crossing[2], crossing[3]), (redundant[2], redundant[3])))
    assert proposal.investigation.best_scene_ids == (crossing[4].image,)
    feedback = record_grounding_feedback(owner, proposal, redundant[4].image,
        (redundant[4].nodes[0],), (redundant[4].nodes[1],), basis=('teacher chooses corroborating scene',))
    assert isinstance(feedback, GroundingFeedback), feedback
    assert feedback.confirmed_query_ids and not feedback.contradicted_query_ids


def test_counterexample_refit_preserves_heldout_and_bounds_requires_readmission_and_consumes_once():
    owner, model, offered, proposal, feedback = answered()
    before = owner.interpretations.get_source(model.evidence_source_id)
    old = get_grounding_model(owner, model)
    eliminated = tuple(q.query for q in old.queries if q.id in feedback.contradicted_query_ids)
    forged = replace(feedback, confirmed_query_ids=('invented-query',))
    assert isinstance(refit_grounding_from_feedback(owner, forged), Unknown)
    updated = refit_grounding_from_feedback(owner, feedback)
    assert isinstance(updated, SceneGroundingModelHandle), updated
    assert updated.group_id == model.group_id and updated.dependency is None
    after = owner.interpretations.get_source(updated.evidence_source_id)
    assert after.metadata['heldout_evidence_ids'] == before.metadata['heldout_evidence_ids']
    assert after.metadata['training_evidence_ids'] == (*before.metadata['training_evidence_ids'], feedback.teaching_record.evidence_source_id)
    for field in ('max_atoms', 'max_patterns', 'max_matches'):
        assert after.metadata[field] == before.metadata[field]
    assert isinstance(get_grounding_model(owner, model), Unknown)
    assert isinstance(get_grounding_model(owner, updated), Unknown)
    assert isinstance(refit_grounding_from_feedback(owner, feedback), Unknown)
    admitted = admit_grounding_model(owner, updated, reason='explicit counterexample fit review')
    learned = get_grounding_model(owner, admitted)
    assert len(learned.training_examples) == 3 and len(learned.validation_examples) == 1
    assert all(q.query not in eliminated for q in learned.queries)
    assert isinstance(record_grounding_feedback(owner, proposal, offered[4].image,
        (offered[4].nodes[0],), (), basis=('duplicate answer',)), Unknown)


@pytest.mark.parametrize('source_kind', ['feedback', 'predictions'])
def test_feedback_or_prediction_mutation_during_refit_rejects_new_version(monkeypatch, source_kind):
    import tensorcode.agent.grounding_investigation as module
    owner, model, _, proposal, feedback = answered()
    original = module.fit_grounding_model
    def corrupting_fit(*args, **kwargs):
        result = original(*args, **kwargs)
        source_id = feedback.evidence_source_id if source_kind == 'feedback' else proposal.evidence_source_id
        owner.interpretations._sources[source_id].metadata['invented'] = True
        return result
    monkeypatch.setattr(module, 'fit_grounding_model', corrupting_fit)
    result = refit_grounding_from_feedback(owner, feedback)
    assert isinstance(result, Unknown) and 'changed during refitting' in result.detail
    group = owner.interpretations.get(model.group_id)
    assert group.selected_id is None and len(group.candidates) == 2 and group.candidates[-1].rejected


def test_reentrant_refit_cannot_consume_feedback_twice(monkeypatch):
    import tensorcode.agent.grounding_investigation as module
    owner, _, _, _, feedback = answered()
    original = module.fit_grounding_model
    nested = []
    def reentrant_fit(*args, **kwargs):
        nested.append(refit_grounding_from_feedback(owner, feedback))
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'fit_grounding_model', reentrant_fit)
    result = refit_grounding_from_feedback(owner, feedback)
    assert isinstance(result, SceneGroundingModelHandle), result
    assert len(nested) == 1 and isinstance(nested[0], Unknown)


def test_reentrant_feedback_cannot_record_two_answers(monkeypatch):
    owner, _, offered, proposal = prepared()
    original = owner.interpretations.get_source
    nested = []
    def get_source(source_id):
        source = original(source_id)
        if source_id == proposal.evidence_source_id and not nested:
            nested.append(None)
            nested[0] = record_grounding_feedback(owner, proposal, offered[4].image,
                (offered[4].nodes[0],), (), basis=('reentrant answer',))
        return source
    monkeypatch.setattr(owner.interpretations, 'get_source', get_source)
    result = record_grounding_feedback(owner, proposal, offered[4].image,
        (offered[4].nodes[0],), (), basis=('original answer',))
    assert isinstance(result, GroundingFeedback), result
    assert len(nested) == 1 and isinstance(nested[0], Unknown)
    assert len(owner._grounding_investigation_feedback) == 1


def test_negative_only_counterexample_is_retained_without_invented_positive_alignment():
    owner, model, offered, proposal = prepared()
    feedback = record_grounding_feedback(owner, proposal, offered[4].image,
        (), (offered[4].nodes[1],), basis=('teacher excludes this reference; no positive identification',))
    assert isinstance(feedback, GroundingFeedback), feedback
    assert feedback.teaching_record.example.positive_refs == ()
    assert feedback.teaching_record.example.negative_refs == (offered[4].nodes[1],)
    original = get_grounding_model(owner, model)
    contradicted = {query.query for query in original.queries if query.id in feedback.contradicted_query_ids}
    assert contradicted
    fitted = refit_grounding_from_feedback(owner, feedback)
    assert isinstance(fitted, SceneGroundingModelHandle), fitted
    admitted = admit_grounding_model(owner, fitted, reason='review negative-only counterexample')
    learned = get_grounding_model(owner, admitted)
    assert all(query.query not in contradicted for query in learned.queries)
    assert learned.training_examples[-1].positive_refs == ()
    assert len(learned.training_examples[-1].negative_refs) == 1
