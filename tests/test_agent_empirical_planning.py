"""Actual Gym execution, composed routes, and fresh evidence dispatch guards."""
from copy import deepcopy
from dataclasses import replace
import pytest
pytest.importorskip('gymnasium')
from eval.learning.empirical_planning_fixture import make_setup, latest, invoke, GOAL
from tensorcode.outcomes import Unknown


@pytest.fixture
def setup():
    result = make_setup()
    try:
        yield result
    finally:
        result.plugin.close()


def propose(s, **kwargs):
    return s.agent.propose_empirical_plan(s.model, latest(s.agent, s.plugin).id, s.calls, GOAL, **kwargs)


def test_compose_fresh_four_step_route_and_verify_goal(setup):
    s = setup
    sequence = s.plugin.sequence
    p = propose(s, max_depth=4)
    assert p.plan.depth == 4
    assert p.plan.selected_call is None  # Several equally short routes.
    assert len(p.plan.first_calls) == 2
    assert s.plugin.sequence == sequence
    assert not tuple(s.agent.store.claims())
    route = (2, 2, 1, 1)
    assert route not in s.episodes
    assert not any(c.effects for c in s.plugin.capabilities())
    for index, action in enumerate(route):
        if index:
            p = propose(s, max_depth=4-index)
        call = next(c for c in p.plan.first_calls if dict(c.args)['action'] == action)
        result = s.agent.execute_empirical_plan(p.id, call=call)
        assert result.receipt.status == 'applied'
        assert result.verification is (index == 3)
        assert result.reason == ('goal_observed' if index == 3 else 'step_observed_replan_required')
        assert result.source_ids
        before = s.plugin.sequence
        replay = s.agent.execute_empirical_plan(p.id, call=call)
        assert replay.reason == 'proposal_already_consumed'
        assert s.plugin.sequence == before
    assert s.plugin.sequence == sequence + 4
    assert not tuple(s.agent.store.claims())


def test_horizon_and_tied_actions_do_not_dispatch(setup):
    s = setup
    before = s.plugin.sequence
    short = propose(s, max_depth=3)
    assert short.plan.depth is None
    assert not short.plan.first_calls
    assert s.agent.execute_empirical_plan(short.id).receipt is None
    tied = propose(s, max_depth=4)
    assert s.agent.execute_empirical_plan(tied.id).receipt is None
    assert s.plugin.sequence == before


def test_changed_world_prevents_dispatch(setup):
    s = setup
    p = propose(s, max_depth=4)
    invoke(s.agent, s.plugin, 'step', action=1)
    before = s.plugin.sequence
    result = s.agent.execute_empirical_plan(p.id, call=p.plan.first_calls[0])
    assert result.receipt.status != 'applied'
    assert isinstance(result.verification, Unknown)
    assert s.plugin.sequence == before


def test_foreign_workspace_cannot_borrow_model_sources(setup):
    from tensorcode.agent import Agent
    s = setup
    foreign = Agent([s.plugin])
    invoke(foreign, s.plugin, 'reset', seed=0)
    with pytest.raises(ValueError, match='retained execution evidence'):
        foreign.propose_empirical_plan(s.model, latest(foreign, s.plugin).id, s.calls, GOAL)


def test_changed_capability_in_final_sensor_callback_prevents_dispatch(setup):
    s = setup
    p = propose(s, max_depth=4)
    observe = s.plugin.observe_evidence
    caps = s.plugin.capabilities()
    observations = 0
    def changed():
        nonlocal observations
        observations += 1
        if observations == 2:
            s.plugin.capabilities = lambda: tuple(replace(c, description='changed contract') for c in caps)
        return observe()
    s.plugin.observe_evidence = changed
    before = s.plugin.sequence
    result = s.agent.execute_empirical_plan(p.id, call=p.plan.first_calls[0])
    assert result.receipt.status != 'applied'
    assert s.plugin.sequence == before


def test_missing_after_observation_does_not_establish_goal(setup):
    s = setup
    p = propose(s, max_depth=4)
    observe = s.plugin.observe_evidence
    observations = 0
    def missing():
        nonlocal observations
        observations += 1
        if observations >= 3:
            raise RuntimeError('sensor disconnected')
        return observe()
    s.plugin.observe_evidence = missing
    result = s.agent.execute_empirical_plan(p.id, call=p.plan.first_calls[0])
    assert result.receipt.status == 'applied'
    assert isinstance(result.verification, Unknown)
    assert result.reason == 'unresolved_observation'


def test_new_counterexample_blocks_reuse_until_refit(setup):
    s = setup
    p = propose(s, max_depth=4)
    step = s.plugin.environment.step
    # An external dynamics change: intended movement now actually moves up.
    s.plugin.environment.step = lambda action: step(3)
    result = s.agent.execute_empirical_plan(p.id, call=p.plan.first_calls[0])
    assert result.receipt.status == 'applied'
    assert result.reason == 'unmodeled_outcome'
    assert isinstance(result.verification, Unknown)
    assert result.observed_state == (0, False, False)
    with pytest.raises(ValueError, match='empirical_counterexample'):
        propose(s, max_depth=4)


def test_final_capability_callback_cannot_unmount_executor(setup):
    s = setup
    p = propose(s, max_depth=4)
    capabilities = s.plugin.capabilities
    calls = 0
    def unmounting():
        nonlocal calls
        calls += 1
        if calls == 3:
            s.agent.plugins = []
        return capabilities()
    s.plugin.capabilities = unmounting
    before = s.plugin.sequence
    result = s.agent.execute_empirical_plan(p.id, call=p.plan.first_calls[0])
    assert result.receipt.status != 'applied'
    assert s.plugin.sequence == before
