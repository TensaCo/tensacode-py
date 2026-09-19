"""Authored miniature dynamics produce executed samples; fitting learns the graph."""
from dataclasses import FrozenInstanceError, replace

import pytest

from tensorcode.agent.plugin import Call
from tensorcode.learning.empirical_dynamics import (
    DynamicsPolicy, StateProjection, fit_dynamics,
)
from tensorcode.learning.experience import Transition
from tensorcode.outcomes import Receipt, Unknown


PROJECTION = StateProjection('counter-state', lambda observation: observation['state'],
                             ('Authored state projection for this executed fixture',))


class World:
    def __init__(self, state, noise):
        self.state = state
        self.noise = noise
    def observe(self):
        return {'state': self.state}
    def execute(self, action):
        self.state += action.arg('step') + self.noise
        return Receipt(action, 'applied')


def samples(*, minority=False):
    rows, train, held = [], [], []
    for noise in ((0, 1) if minority else (0,)):
        for index in range(3):
            identifier = f'{noise}:{index}'
            world = World(0, noise)
            before = world.observe()
            action = Call('counter', 'advance', (('step', 1),))
            receipt = world.execute(action)
            rows.append(Transition(identifier, 'plugin:counter', (identifier + ':before', identifier + ':after'),
                        before, action, world.observe(), receipt))
            (train if index < 2 else held).append(identifier)
    return tuple(rows), tuple(train), tuple(held)


def fit(rows=None, train=None, held=None, **kwargs):
    if rows is None:
        rows, train, held = samples()
    return fit_dynamics(rows, projection=PROJECTION, train_attempt_ids=train,
                        evaluation_attempt_ids=held, **kwargs)


def test_edges_are_fitted_from_executions_with_separate_support_for_every_successor():
    rows, train, held = samples(minority=True)
    model = fit(rows, train, held)
    prediction = model.predict(0, rows[0].action)
    assert [outcome.state for outcome in prediction.outcomes] == [1, 2]
    assert prediction.model_id == model.id and model.revision == 0
    for outcome in prediction.outcomes:
        assert len(outcome.training_attempt_ids) == 2
        assert len(outcome.evaluation_attempt_ids) == 1
        assert len(outcome.source_ids) == 6
    assert model.states == (0, 1, 2) and model.calls == (rows[0].action,)
    assert model.validate_transitions(reversed(rows)) is True


def test_training_only_minority_is_preserved_and_blocks_entire_edge():
    rows, train, held = samples(minority=True)
    rows = tuple(row for row in rows if row.attempt_id != held[-1])
    model = fit(rows, train, held[:-1])
    assert [outcome.state for outcome in model.edges[0].outcomes] == [1, 2]
    assert model.edges[0].outcomes[-1].evaluation_attempt_ids == ()
    assert not model.edges[0].eligible
    assert model.predict(0, rows[0].action).reason == 'insufficient_outcome_support'


def test_evaluation_only_outcome_is_not_silently_removed():
    rows, train, held = samples(minority=True)
    rows = tuple(row for row in rows if row.attempt_id not in train[-2:])
    model = fit(rows, train[:-2], held)
    assert model.edges[0].outcomes[-1].training_attempt_ids == ()
    assert isinstance(model.predict(0, rows[0].action), Unknown)


def test_bool_and_integer_states_and_action_arguments_do_not_alias():
    rows, train, held = samples()
    modified = tuple(replace(row, before={'state': False}, after={'state': True}) for row in rows)
    model = fit(modified, train, held)
    assert model.states == (False, True)
    assert not isinstance(model.predict(False, rows[0].action), Unknown)
    assert isinstance(model.predict(0, rows[0].action), Unknown)
    assert isinstance(model.predict(False, replace(rows[0].action, args=(('step', True),))), Unknown)


def test_unknown_states_and_calls_have_no_default_edge():
    rows, train, held = samples()
    model = fit(rows, train, held)
    for state, call in ((9, rows[0].action), (0, replace(rows[0].action, capability='other')),
                        (0, replace(rows[0].action, args=(('step', 2),)))):
        assert isinstance(model.predict(state, call), Unknown)


@pytest.mark.parametrize('state', [float('nan'), float('inf'), ('state', float('-inf')), {'state': 0}, [0], Unknown('missing')])
def test_invalid_or_unknown_projection_states_cannot_be_fitted(state):
    rows, train, held = samples()
    projection = StateProjection('invalid', lambda _: state, ('Authored invalid fixture',))
    with pytest.raises(ValueError, match='finite primitive'):
        fit_dynamics(rows, projection=projection, train_attempt_ids=train, evaluation_attempt_ids=held)


def test_recursive_tuple_state_including_terminal_marker_is_explicit():
    rows, train, held = samples()
    projection = StateProjection('terminal-aware', lambda observed: ('counter', observed['state'], observed['state'] == 1),
                                 ('Authored terminal definition for this fixture',))
    model = fit_dynamics(rows, projection=projection, train_attempt_ids=train, evaluation_attempt_ids=held)
    assert model.predict(('counter', 0, False), rows[0].action).outcomes[0].state == ('counter', 1, True)
    assert isinstance(model.predict(('counter', 0, 0), rows[0].action), Unknown)


def test_split_and_source_identity_validation():
    rows, train, held = samples()
    for training, evaluation in ((train + held, held), (train[:1], held), (train + train[:1], held), ((), train + held)):
        with pytest.raises(ValueError):
            fit(rows, training, evaluation)
    with pytest.raises(ValueError, match='duplicate attempt'):
        fit(rows + rows[:1], train, held)
    with pytest.raises(ValueError, match='independent observation'):
        fit((rows[0], replace(rows[1], source_ids=rows[0].source_ids), rows[2]), train, held)
    with pytest.raises(ValueError, match='applied receipts'):
        fit((replace(rows[0], receipt=Receipt(rows[0].action, 'rejected')), *rows[1:]), train, held)


def test_retained_source_replay_rejects_forged_fit_or_modified_raw_evidence():
    rows, train, held = samples()
    altered = tuple(replace(row, after={'state': 99}) for row in rows)
    forged = fit(altered, train, held)
    assert isinstance(forged.validate_transitions(rows), Unknown)
    genuine = fit(rows, train, held)
    assert isinstance(genuine.validate_transitions(altered), Unknown)
    assert isinstance(genuine.validate_transitions(rows[:-1]), Unknown)
    assert isinstance(genuine.validate_transitions(rows + rows[:1]), Unknown)


def test_replay_recomputes_projection_and_aggregation_instead_of_trusting_cached_edges():
    rows, train, held = samples()
    model = fit(rows, train, held)
    object.__setattr__(model, '_edges', (replace(model.edges[0], eligible=False),))
    assert isinstance(model.validate_transitions(rows), Unknown)
    offset = [0]
    projection = StateProjection('mutable-projection', lambda observed: observed['state'] + offset[0],
                                 ('Authored callback with mutable state for adversarial test',))
    model = fit_dynamics(rows, projection=projection, train_attempt_ids=train, evaluation_attempt_ids=held)
    offset[0] = 1
    assert isinstance(model.validate_transitions(rows), Unknown)


def test_detached_snapshots_and_frozen_model_preserve_fit():
    rows, train, held = samples()
    action = replace(rows[0].action, args=(('step', {'value': 1}),))
    rows = tuple(replace(row, action=action, receipt=Receipt(action, 'applied')) for row in rows)
    model = fit(rows, train, held)
    action.args[0][1]['value'] = 2
    model.calls[0].args[0][1]['value'] = 3
    model.examples[0].call.args[0][1]['value'] = 4
    model.edges[0].call.args[0][1]['value'] = 5
    assert model.calls[0].args[0][1] == {'value': 1}
    with pytest.raises(FrozenInstanceError):
        model._id = 'replacement'


@pytest.mark.parametrize('threshold', [0, -1, True, 1.5])
def test_support_thresholds_are_strict_positive_integers(threshold):
    with pytest.raises(ValueError):
        DynamicsPolicy(min_training_support=threshold)
    with pytest.raises(ValueError):
        DynamicsPolicy(min_evaluation_support=threshold)


def test_later_actual_counterexample_blocks_known_edge_until_refit():
    rows, train, held = samples()
    model = fit(rows, train, held)
    world = World(0, 1)
    before = world.observe()
    receipt = world.execute(rows[0].action)
    later = Transition('later', 'plugin:counter', ('later:before', 'later:after'),
                       before, rows[0].action, world.observe(), receipt)
    result = model.validate_transitions((*rows, later))
    assert isinstance(result, Unknown) and result.reason == 'empirical_counterexample'
    assert result.detail == 'later'
    assert model.revision == 0
    assert [outcome.state for outcome in model.edges[0].outcomes] == [1]
    # A familiar successor does not invalidate fitted evidence or update its counts.
    familiar = replace(later, after={'state': 1})
    assert model.validate_transitions((*rows, familiar)) is True
    assert len(model.edges[0].outcomes[0].training_attempt_ids) == 2


def test_later_unknown_states_actions_and_other_providers_do_not_generalize_edges():
    rows, train, held = samples()
    model = fit(rows, train, held)
    original = rows[0]
    later = replace(original, attempt_id='later', source_ids=('later:before', 'later:after'),
                    before={'state': 17}, after={'state': 99})
    assert model.validate_transitions((*rows, later)) is True
    action = replace(original.action, capability='unknown')
    later = replace(later, before={'state': 0}, action=action, receipt=Receipt(action, 'applied'))
    assert model.validate_transitions((*rows, later)) is True
    later = replace(later, provider='plugin:other', action=original.action,
                    receipt=Receipt(original.action, 'applied'))
    assert model.validate_transitions((*rows, later)) is True


def test_unmodeled_reset_is_skipped_before_projection_but_known_calls_fail_closed():
    rows, train, held = samples()
    model = fit(rows, train, held)
    reset = Call('counter', 'reset', ())
    extra = Transition('initial-reset', 'plugin:counter', ('reset:before', 'reset:after'),
                       None, reset, {'state': 0}, Receipt(reset, 'applied'))
    assert model.validate_transitions((extra, *rows)) is True
    known = replace(extra, action=rows[0].action, receipt=Receipt(rows[0].action, 'applied'))
    result = model.validate_transitions((known, *rows))
    assert isinstance(result, Unknown) and result.reason == 'dynamics_evidence_mismatch'
