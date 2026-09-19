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
