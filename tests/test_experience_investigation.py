"""Fresh executions distinguish applicability of empirically learned models."""
import pytest

from tensorcode.agent.experience_investigation import ModelApplicability, execute, propose
from tensorcode.outcomes import Unknown
from experience_investigation_fixtures import PROJECTION, bindings, make_setup, unknown_model


def proposal(setup, hypotheses=None):
    return propose(setup.agent, setup.group_id, bindings(setup) if hypotheses is None else hypotheses,
                   setup.before_source_id, setup.calls)


def test_fresh_probe_distinguishes_learned_context_without_selecting_or_suspending():
    setup = make_setup()
    before = setup.agent.interpretations.get(setup.group_id)
    count = len(setup.plugin.executions)
    plan = proposal(setup)
    assert plan.selected_call == setup.calls[0]
    assert len(setup.plugin.executions) == count
    assert [row.prediction.outcome for row in plan.probes[0].predictions] == [1, 2]
    for row in plan.probes[0].predictions:
        assert row.prediction.evidence.training_attempt_ids
        assert row.prediction.evidence.evaluation_attempt_ids
        assert set(row.prediction.evidence.training_attempt_ids).isdisjoint(row.prediction.evidence.evaluation_attempt_ids)
        assert row.prediction.projection_provenance == PROJECTION.provenance
    result = execute(setup.agent, plan.id)
    assert result.receipt.status == 'applied'
    assert len(setup.plugin.executions) == count + 1
    assert setup.plugin.observe_evidence() == {'lamp': 2}
    assert [assessment.status for assessment in result.assessments] == ['contradicted', 'confirmed']
    assert result.supported_candidate_id == setup.candidate_ids[1]
    assert setup.agent.interpretations.get(setup.group_id) == before
    assert [model.revision for model in setup.models] == [0, 0]
    assert not tuple(setup.agent.store.claims()) and not tuple(setup.agent.store.propositions())
    assert not setup.agent.turns
    assert all(not capability.effects for capability in setup.plugin.capabilities())
    sources = tuple(setup.agent.interpretations.get_source(sid) for sid in result.source_ids)
    after = tuple(source for source in sources if source.metadata.get('stage') == 'after_action')
    assert len(after) == 1 and after[0].payload == {'lamp': 2}
    assert after[0].metadata['receipt'].status == 'applied'
    training_sources = {sid for model in setup.models for example in model.examples for sid in example.source_ids}
    assert training_sources.isdisjoint(result.source_ids)
    assert setup.agent.interpretations.get_source(result.record_source_id).modality == 'assessment'


def test_equal_learned_predictions_do_not_invent_a_discriminating_probe():
    setup = make_setup()
    hypotheses = tuple(ModelApplicability(candidate, setup.models[0], ('Authored same-model applicability',))
                       for candidate in setup.candidate_ids)
    count = len(setup.plugin.executions)
    plan = proposal(setup, hypotheses)
    assert plan.selected_call is None and not plan.best_calls
    assert plan.reason == 'no_supported_discriminating_probe'
    result = execute(setup.agent, plan.id)
    assert result.receipt is None and result.supported_candidate_id is None
    assert len(setup.plugin.executions) == count


def test_unknown_rival_cannot_be_eliminated_by_other_models_confirmation():
    setup = make_setup()
    candidate = setup.agent.interpretations.propose(setup.group_id, {'context': 'baseline-only'},
                provenance=('Authored third context with incomplete action experience',))
    model = unknown_model(setup)
    hypotheses = (*bindings(setup), ModelApplicability(candidate.id, model, ('Authored baseline-only applicability',)))
    plan = proposal(setup, hypotheses)
    assert isinstance(plan.probes[0].predictions[-1].prediction, Unknown)
    result = execute(setup.agent, plan.id)
    assert [assessment.status for assessment in result.assessments] == ['contradicted', 'confirmed', 'unresolved']
    assert result.supported_candidate_id is None
    assert result.reason == 'unresolved_applicability'
    assert model.revision == 0


def test_unexpected_outcome_contradicts_all_supplied_contexts_without_global_revision():
    setup = make_setup(mode=3)
    result = execute(setup.agent, proposal(setup).id)
    assert result.receipt.status == 'applied'
    assert all(assessment.status == 'contradicted' for assessment in result.assessments)
    assert result.supported_candidate_id is None
    assert result.reason == 'all_applicability_hypotheses_contradicted'
    assert [model.revision for model in setup.models] == [0, 0]


def test_tied_probes_require_explicit_choice_from_retained_best_calls():
    setup = make_setup(tie=True)
    count = len(setup.plugin.executions)
    plan = proposal(setup)
    assert plan.selected_call is None and plan.best_calls == setup.calls
    assert plan.reason == 'probe_choice_required'
    result = execute(setup.agent, plan.id, call=setup.calls[1])
    assert result.receipt.action == setup.calls[1]
    assert len(setup.plugin.executions) == count + 1
    assert result.supported_candidate_id == setup.candidate_ids[1]


def test_hypotheses_cover_every_candidate_exactly_once():
    setup = make_setup()
    hypotheses = bindings(setup)
    for incomplete in (hypotheses[:1], hypotheses + hypotheses[:1]):
        with pytest.raises(ValueError):
            proposal(setup, incomplete)
    assert setup.agent.interpretations.get(setup.group_id).selected_id is None


@pytest.mark.parametrize('observed', [1, 2])
def test_empirical_minority_outcome_remains_viable_in_noisy_context(observed):
    from experience_investigation_fixtures import make_noisy_setup
    setup = make_noisy_setup(mode=observed)
    plan = proposal(setup)
    assert plan.selected_call == setup.calls[0]
    first, second = plan.probes[0].predictions
    assert first.prediction.outcome == 1  # Dominant point label is not exhaustive.
    outcomes = {support.outcome: support for support in first.supported_outcomes}
    assert set(outcomes) == {1, 2}
    assert (outcomes[1].training_count, outcomes[1].evaluation_count) == (30, 6)
    assert (outcomes[2].training_count, outcomes[2].evaluation_count) == (10, 2)
    assert {support.outcome for support in second.supported_outcomes} == {2}
    assert all(support.source_ids for support in first.supported_outcomes)
    result = execute(setup.agent, plan.id)
    assert result.receipt.status == 'applied'
    assert result.assessments[0].status == 'confirmed'
    if observed == 2:
        assert result.assessments[1].status == 'confirmed'
        assert result.supported_candidate_id is None
        assert result.reason == 'unresolved_applicability'
    else:
        assert result.assessments[1].status == 'contradicted'
        assert result.supported_candidate_id == setup.candidate_ids[0]
    assert setup.agent.interpretations.get(setup.group_id).selected_id is None
    assert [model.revision for model in setup.models] == [0, 0]


def test_equal_empirical_outcome_sets_do_not_offer_discrimination():
    from experience_investigation_fixtures import make_noisy_setup
    setup = make_noisy_setup()
    hypotheses = tuple(ModelApplicability(candidate, setup.models[0], ('Authored same-noisy-context correspondence',))
                       for candidate in setup.candidate_ids)
    count = len(setup.plugin.executions)
    plan = proposal(setup, hypotheses)
    assert all({support.outcome for support in row.supported_outcomes} == {1, 2}
               for row in plan.probes[0].predictions)
    assert plan.selected_call is None and not plan.best_calls
    result = execute(setup.agent, plan.id)
    assert result.receipt is None and result.supported_candidate_id is None
    assert len(setup.plugin.executions) == count
