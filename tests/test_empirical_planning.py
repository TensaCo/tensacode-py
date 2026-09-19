"""Pure search tests over fitted authored transition fixtures, not perception tests."""
from copy import deepcopy

import pytest

from tensorcode.agent.empirical_planning import plan
from tensorcode.agent.plugin import Call
from tensorcode.learning.empirical_dynamics import StateProjection, fit_dynamics
from tensorcode.learning.experience import Transition
from tensorcode.outcomes import Receipt, Unknown


LEFT = Call('fixture', 'move', (('direction', 'left'),))
RIGHT = Call('fixture', 'move', (('direction', 'right'),))
PROJECTION = StateProjection('fixture identity', lambda raw: raw, ('Authored symbolic state fixture',))


def fitted(edges):
    """Supplied observations exercise induction and search; no live-world claim."""
    rows, training, evaluation = [], [], []
    for before, call, after in edges:
        for sample in range(3):
            attempt = f'authored:{len(rows)}'
            rows.append(Transition(attempt, 'plugin:fixture', (attempt + ':before', attempt + ':after'),
                                   deepcopy(before), call, deepcopy(after), Receipt(call, 'applied')))
            (training if sample < 2 else evaluation).append(attempt)
    return fit_dynamics(rows, projection=PROJECTION, train_attempt_ids=training, evaluation_attempt_ids=evaluation)


def test_composes_separately_fitted_edges_into_new_two_step_policy():
    model = fitted([(0, RIGHT, 1), (1, LEFT, 2)])
    result = plan(model, 0, 2, (LEFT, RIGHT))
    assert result.supported and result.depth == 2 and result.selected_call == RIGHT
    assert result.first_calls == (RIGHT,)
    middle = next(node for node in result.nodes if node.state == 1)
    assert middle.rank == 1 and middle.calls == (LEFT,)
    assert any(isinstance(edge.prediction, Unknown) for edge in result.edges)
    supported = [edge for edge in result.edges if not isinstance(edge.prediction, Unknown)]
    assert all(outcome.source_ids for edge in supported for outcome in edge.prediction.outcomes)
    assert not any(edge.state == 2 for edge in result.edges)  # no goal expansion


def test_all_observed_branches_must_have_supported_continuations():
    successful = fitted([(0, RIGHT, 1), (0, RIGHT, 2), (1, LEFT, 3), (2, RIGHT, 3)])
    result = plan(successful, 0, 3, (LEFT, RIGHT))
    assert result.depth == 2 and result.selected_call == RIGHT
    trap = fitted([(0, RIGHT, 1), (0, RIGHT, 2), (1, LEFT, 3), (2, RIGHT, 2)])
    unresolved = plan(trap, 0, 3, (LEFT, RIGHT))
    assert not unresolved.supported and unresolved.depth is None and not unresolved.first_calls
    assert unresolved.reason == 'no_supported_policy_within_bounds'


def test_cycle_cannot_bootstrap_a_goal_and_exit_provides_finite_rank():
    cycle = fitted([(0, LEFT, 1), (1, RIGHT, 0)])
    assert not plan(cycle, 0, 2, (LEFT, RIGHT)).supported
    exit_model = fitted([(0, LEFT, 1), (1, RIGHT, 0), (1, LEFT, 2)])
    assert plan(exit_model, 0, 2, (LEFT, RIGHT)).depth == 2
    # A stochastic self-loop has no finite worst-case depth despite a goal branch.
    self_loop = fitted([(0, RIGHT, 0), (0, RIGHT, 2)])
    assert not plan(self_loop, 0, 2, (RIGHT,), max_depth=20).supported


def test_shortest_tied_first_actions_are_retained_without_order_authority():
    model = fitted([(0, RIGHT, 1), (0, LEFT, 1), (1, RIGHT, 2)])
    result = plan(model, 0, 2, (RIGHT, LEFT))
    assert result.depth == 2 and result.first_calls == (RIGHT, LEFT)
    assert result.selected_call is None and result.reason == 'first_action_choice_required'
    reversed_result = plan(model, 0, 2, (LEFT, RIGHT))
    assert reversed_result.first_calls == (LEFT, RIGHT) and reversed_result.selected_call is None


def test_bounds_abstain_without_hiding_an_unexamined_tie():
    model = fitted([(0, RIGHT, 1), (0, LEFT, 1), (1, RIGHT, 2)])
    shallow = plan(model, 0, 2, (RIGHT, LEFT), max_depth=1)
    assert not shallow.supported and shallow.depth_limited
    for limits in ({'max_states': 1}, {'max_edges': 1}, {'max_edges': 0}):
        result = plan(model, 0, 2, (RIGHT, LEFT), **limits)
        assert result.budget_exhausted and not result.supported
        assert result.reason == 'graph_budget_exhausted' and not result.first_calls
        assert result.prediction_count <= limits.get('max_edges', 100000)
        assert result.state_count <= limits.get('max_states', 10000)


def test_goal_match_is_conditional_on_supplied_state_and_uses_no_edges():
    model = fitted([(0, RIGHT, 1)])
    result = plan(model, 0, 0, (RIGHT,), max_depth=0, max_edges=0)
    assert result.supported and result.depth == 0
    assert result.reason == 'supplied_state_matches_goal'
    assert not result.edges and not result.first_calls and result.selected_call is None
    # Typed identity: a numeric observation does not establish a Boolean goal.
    boolean = plan(model, 0, False, (RIGHT,))
    assert not boolean.supported


def test_unsupported_successor_bundle_and_bad_bounds_do_not_create_plans():
    model = fitted([(0, RIGHT, 1)])
    assert not plan(model, 0, 1, (LEFT,)).supported
    with pytest.raises(ValueError):
        plan(model, 0, 1, (RIGHT,), max_depth=True)
    with pytest.raises(ValueError):
        plan(model, 0, 1, (RIGHT, RIGHT))


@pytest.mark.parametrize('invalid', [Unknown('unobserved'), float('nan'), float('inf'),
                                     -float('inf'), ('nested', float('nan')), {'state': 0}])
def test_invalid_supplied_states_cannot_establish_even_an_identical_goal(invalid):
    model = fitted([(0, RIGHT, 1)])
    for current, goal in ((invalid, invalid), (0, invalid), (invalid, 0)):
        with pytest.raises(ValueError, match='states must be finite'):
            plan(model, current, goal, (RIGHT,))
