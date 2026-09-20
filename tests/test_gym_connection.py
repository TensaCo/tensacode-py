"""Actual Gymnasium transports: no learned task performance is claimed."""
import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
from examples.general_agent.gym_connection import GymPlugin
from examples.general_agent.plugins import mount
from tensorcode.agent.plugin import Call


def invoke(plugin, capability, **args):
    return plugin.execute(Call(plugin.name, capability, tuple(args.items())))


@pytest.mark.parametrize("environment_id,action", [("CartPole-v1", 0), ("Pendulum-v1", np.array([0.0], dtype=np.float32))])
def test_real_environments_require_reset_and_retain_observations(environment_id, action):
    mounted = mount("gym:" + environment_id)
    plugin = mounted.plugin
    try:
        assert mounted.descriptor()["status"] == "connected"
        assert mounted.descriptor()["kind"] == "gym"
        assert mounted.descriptor()["preview"] is None
        assert invoke(plugin, "step", action=action).status == "rejected"
        assert invoke(plugin, "reset", seed=17).status == "applied"
        assert invoke(plugin, "step", action=action).status == "applied"
        observed = plugin.observe()
        assert observed["sequence"] == 2
        transition = observed["transition"]
        assert plugin.environment.observation_space.contains(transition["observation"])
        assert transition["operation"] == "step"
        assert transition["reward"] is not None
        transition["observation"][:] = 999
        assert not np.all(plugin.observe()["transition"]["observation"] == 999)
        assert plugin.screenshot() is None  # absent render mode means no invented preview
        assert tuple(plugin.perceive()) == ()
        assert all(not cap.effects for cap in plugin.capabilities())
    finally:
        mounted.close()
    assert invoke(plugin, "reset", seed=None).status == "rejected"


def test_real_truncation_requires_explicit_reset():
    plugin = GymPlugin(gym.make("CartPole-v1", max_episode_steps=1))
    try:
        assert invoke(plugin, "reset", seed=2).status == "applied"
        assert invoke(plugin, "step", action=1).status == "applied"
        assert plugin.observe()["transition"]["truncated"] is True
        assert plugin.observe()["transition"]["terminated"] is False
        assert invoke(plugin, "step", action=0).status == "rejected"
        assert plugin.observe()["sequence"] == 2
        assert invoke(plugin, "reset", seed=2).status == "applied"
        assert not plugin.needs_reset
    finally:
        plugin.close()


def test_invalid_actions_rejected_before_environment_mutates():
    plugin = GymPlugin.from_id("CartPole-v1")
    try:
        invoke(plugin, "reset", seed=1)
        for action in (9, -1, "left", None):
            assert invoke(plugin, "step", action=action).status == "rejected"
        assert plugin.sequence == 1
    finally:
        plugin.close()


def test_uncertain_mutation_requires_explicit_reset():
    class Failing(gym.Wrapper):
        def step(self, action):
            self.env.step(action)
            raise TimeoutError("reply lost after advancing")
    plugin = GymPlugin(Failing(gym.make("CartPole-v1")))
    try:
        invoke(plugin, "reset", seed=1)
        assert invoke(plugin, "step", action=0).status == "indeterminate"
        assert plugin.observe()["transition"] is None
        assert invoke(plugin, "step", action=0).status == "rejected"
    finally:
        plugin.close()


def test_invalid_environment_never_becomes_connected():
    with pytest.raises(gym.error.Error):
        mount("gym:TensorcodeEnvironmentDoesNotExist-v999")


def test_agent_retains_real_transitions_without_asserting_their_meaning():
    from tensorcode.agent import Agent
    from tensorcode.runtime import use

    plugin = GymPlugin.from_id('CartPole-v1')
    agent = Agent([plugin])
    events = []
    try:
        with use(agent.runtime):
            for name, args in [('reset', {'seed': 27}), ('step', {'action': 1})]:
                cap = next(c for c in plugin.capabilities() if c.name == name)
                assert agent._invoke(plugin, cap, args, events).status == 'applied'
        sources = agent.interpretations.sources()
        after = [s for s in sources if s.metadata.get('stage') == 'after_action']
        assert len(after) == 2
        reset, step = after
        assert reset.payload['transition']['operation'] == 'reset'
        assert step.payload['transition']['operation'] == 'step'
        assert step.payload['transition']['reward'] == 1.0
        assert step.metadata['attempt_id'] != reset.metadata['attempt_id']
        old = reset.payload['transition']['observation'].copy()
        step.payload['transition']['observation'][:] = 999
        assert np.array_equal(agent.interpretations.get_source(reset.id).payload['transition']['observation'], old)
        assert not np.all(agent.interpretations.get_source(step.id).payload['transition']['observation'] == 999)
        assert not list(agent.store.claims())
        assert not list(agent.store.propositions())
    finally:
        plugin.close()


def test_learns_supported_transition_rules_from_retained_real_gym_evidence():
    from tensorcode.agent import Agent
    from tensorcode.learning.experience import Projection, extract_transitions, fit_transitions
    from tensorcode.outcomes import Unknown
    from tensorcode.runtime import use

    plugin = GymPlugin.from_id('CartPole-v1')
    agent = Agent([plugin])
    capabilities = {c.name: c for c in plugin.capabilities()}
    events = []
    try:
        with use(agent.runtime):
            # Independent seeded starts; the projection defines the measured
            # outcome but never supplies an action-to-outcome rule.
            for index in range(48):
                seed = index if index < 32 else 1000 + index
                assert agent._invoke(plugin, capabilities['reset'], {'seed': seed}, events).status == 'applied'
                assert agent._invoke(plugin, capabilities['step'], {'action': index % 2}, events).status == 'applied'
        batch = extract_transitions(agent.interpretations.sources(), provider='plugin:' + plugin.name)
        assert not batch.exclusions
        rows = tuple(row for row in batch.transitions if row.action.capability == 'step')
        projection = Projection(
            name='cart-velocity-sign',
            features=lambda before, action: {'selected_action': dict(action.args)['action']},
            outcome=lambda before, action, after: 'positive' if after['transition']['observation'][1] > 0 else 'nonpositive',
            provenance=('Authored measurement: sign of Gym CartPole cart velocity; selected action is an input feature.',),
        )
        model = fit_transitions(rows, projection=projection,
                                train_attempt_ids=[r.attempt_id for r in rows[:32]],
                                evaluation_attempt_ids=[r.attempt_id for r in rows[32:]])
        assert model.evaluation.accuracy == 1.0
        assert model.evaluation.coverage >= .5
        validated_ids = {s for evidence in model.evidence for s in evidence.source_ids}
        assert validated_ids <= {s.id for s in agent.interpretations.sources()}
        predicted = 0
        with use(agent.runtime):
            for action in (0, 1):
                agent._invoke(plugin, capabilities['reset'], {'seed': 8000 + action}, events)
                before = plugin.observe()
                call = Call(plugin.name, 'step', (('action', action),))
                prediction = model.predict(before, call)
                agent._invoke(plugin, capabilities['step'], {'action': action}, events)
                if not isinstance(prediction, Unknown):
                    predicted += 1
                    assert prediction.outcome == projection.outcome(before, call, plugin.observe())
        assert predicted >= 1  # uncovered rules remain Unknown, not a majority fallback
        assert not list(agent.store.claims())
        assert not list(agent.store.propositions())
    finally:
        plugin.close()


@pytest.mark.parametrize('capability,arguments', [
    ('reset', (('seed', 1), ('seed', 2))),
    ('step', (('action', 0), ('action', 1))),
])
def test_duplicate_argument_names_cannot_reset_or_step_the_environment(capability, arguments):
    plugin = GymPlugin.from_id('CartPole-v1')
    try:
        assert invoke(plugin, 'reset', seed=17).status == 'applied'
        original = np.array(plugin.environment.unwrapped.state, copy=True)
        receipt = plugin.execute(Call(plugin.name, capability, arguments))
        assert receipt.status == 'rejected' and 'unique' in receipt.error
        assert plugin.sequence == 1
        assert not plugin.needs_reset
        assert np.array_equal(plugin.environment.unwrapped.state, original)
    finally:
        plugin.close()
